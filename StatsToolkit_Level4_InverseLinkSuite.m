function ResultsTable = StatsToolkit_Level4_InverseLinkSuite(mdl)
% STATSTOOLKIT_LEVEL4_INVERSELINKSUITE Automated extraction of Marginal Means.
%
% Accepts a single model object (LME or GLME).
% Extracts population-level predictions already transformed back to the 
% original response scale.
%
% Note:
% For GLMMs with a 'log' link function, it applies an analytical correction 
% for Jensen's inequality bias: E[Y] approx exp(X*beta + 0.5*sigma^2_re)
%
% Syntax:
%   ResultsTable = StatsToolkit_Level0_InverseLinkSuite(mdl)
%
% Inputs:
%   mdl - Object of class LinearMixedModel or GeneralizedLinearMixedModel.
%
% Outputs:
%   ResultsTable - Table containing unique fixed-effect combinations and 
%                  their respective marginal predictions (Mean, Lower CI, Upper CI).
%
% Dependencies: Statistics and Machine Learning Toolbox (R2025a)

    arguments
        mdl {mustBeA(mdl, ["LinearMixedModel", "GeneralizedLinearMixedModel"])}
    end

    %% =========================================================
    % 1. Model Metadata Extraction
    %% =========================================================
    allVars = string(mdl.PredictorNames);
    groupingVars = string(mdl.Formula.GroupingVariableNames);
    
    % Explicitly subtract random effects from the predictors list
    fixedVars = setdiff(allVars, groupingVars, 'stable');
    
    tblData = mdl.Variables;
    respVar = char(mdl.ResponseName); % Cast to char for robust dynamic field naming

    %% =========================================================
    % 2. Prediction Grid Construction
    %% =========================================================
    % Creates a grid based on actual unique combinations present in the data
    predictionTable = unique(tblData(:, fixedVars), 'rows');
    nComb = height(predictionTable);

    %% =========================================================
    % 3. Add Placeholders for Random Variables
    %% =========================================================
    for g = 1:numel(groupingVars)
        % Correct indexing for native R2025a string array
        varName = char(groupingVars(g));     
        predictionTable.(varName) = repmat(tblData.(varName)(1), nComb, 1);
    end

    %% =========================================================
    % 4. Prediction (Fixed Effects Only)
    %% =========================================================
    % Conditional = false evaluates fixed effects only. For GLMEs, MATLAB 
    % automatically applies the inverse link function to the output.
    [yHat, yCI] = predict(mdl, predictionTable, 'Conditional', false);

    %% =========================================================
    % 5. Analytical Correction (When Applicable)
    %% =========================================================
    appliedCorrection = false;
    
    if isa(mdl, 'GeneralizedLinearMixedModel')
        linkName = mdl.Link.Name;
        
        if strcmp(linkName, 'log')
            try
                % Robust method to extract covariance matrix (D)
                [D, ~] = covarianceParameters(mdl);
                varRand = 0;
                for i = 1:numel(D)
                    % Element (1,1) represents the intercept variance
                    varRand = varRand + D{i}(1,1); 
                end 
                
                % Jensen's inequality correction factor
                correctionFactor = exp(0.5 * varRand);
                
                % Apply correction to estimates and intervals
                yHat = yHat * correctionFactor;
                yCI  = yCI  * correctionFactor;
                
                appliedCorrection = true;
                
            catch
                warning('Could not apply analytical correction (complex random structure detected).');
            end
        end
    end

    %% =========================================================
    % 6. Output Organization
    %% =========================================================
    outCols = table();
    outCols.(sprintf('%s_Mean', respVar))  = yHat;
    outCols.(sprintf('%s_Lower', respVar)) = yCI(:, 1);
    outCols.(sprintf('%s_Upper', respVar)) = yCI(:, 2);

    %% =========================================================
    % 7. User Feedback
    %% =========================================================
    if isa(mdl, 'GeneralizedLinearMixedModel')
        if strcmp(mdl.Link.Name, 'log')
            if appliedCorrection
                fprintf(['ℹ Model (%s) [GLME]: Predictions on original scale ' ...
                         'with Jensen correction (link = log).\n'], respVar);
            else
                fprintf(['⚠ Model (%s) [GLME]: Log link detected, but correction ' ...
                         'was not applied (complex random structure).\n'], respVar);
            end
        else
            fprintf(['ℹ Model (%s) [GLME]: Predictions on original scale ' ...
                     '(link = %s inverted, no correction needed).\n'], ...
                     respVar, mdl.Link.Name);
        end
    else
        fprintf('ℹ Model (%s) [LME]: Values on standard continuous scale.\n', respVar);
    end

    %% =========================================================
    % 8. Final Table Assembly
    %% =========================================================
    ResultsTable = [predictionTable(:, fixedVars), outCols];
    fprintf('✅ Success: Prediction table generated for %d conditions.\n', nComb);
end