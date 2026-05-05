function results = StatsToolkit_Level1_SelectDistribution(tbl)
% STATSTOOLKIT_LEVEL1_SELECTDISTRIBUTION Fits candidate distributions and selects the best model via AIC.
%
% Syntax:
%   results = StatsToolkit_Level1_SelectDistribution(tbl)
%
% Inputs:
%   tbl - Data table. The expected structure is:
%         [RandomFactor, FixedFactor1, ..., FixedFactorN, Response]
%
% Outputs:
%   results - Structure containing the optimized model and diagnostics:
%       .BestModel    : Fitted LinearMixedModel or GeneralizedLinearMixedModel object.
%       .Distribution : String indicating the chosen distribution ('normal', 'poisson', etc.).
%       .modelType    : String ('lme' or 'glme').
%       .Diagnostics  : Structure with overdispersion and zero-inflation tests (if applicable).
%
% Dependencies: Statistics and Machine Learning Toolbox (R2025a)

    % ==========================================
    % 1. Automatic Formula Construction
    % ==========================================
    varnames = tbl.Properties.VariableNames;
    response = varnames{end};
    randomFactor = varnames{1};
    
    % Builds the fixed component assuming full interaction (*) among predictors
    fixedPart = strjoin(varnames(2:end-1), '*');
    formula = sprintf('%s ~ %s + (1|%s)', response, fixedPart, randomFactor);
    
    y = tbl{:, response};
    y_valid = y(~isnan(y)); % Isolates valid data for structural tests
    
    % ==========================================
    % 2. Response Variable Structural Filters
    % ==========================================
    isInteger  = all(mod(y_valid, 1) == 0);
    isNonNeg   = all(y_valid >= 0);
    isPositive = all(y_valid > 0);
    isBinary   = all(ismember(y_valid, [0 1]));
    meanY      = mean(y_valid);
    
    candidates = {};
    
    % Defines the search space based on data format
    if ~isBinary
        % Avoids Normal distribution for small discrete counts with strong Poisson bias
        if ~(isInteger && isNonNeg && meanY <= 10)
            candidates{end+1} = struct('Dist', 'normal', 'Link', 'identity');
        end
    end
    
    if isInteger && isNonNeg && ~isBinary
        candidates{end+1} = struct('Dist', 'poisson', 'Link', 'log');
    end
    
    if isBinary
        candidates{end+1} = struct('Dist', 'binomial', 'Link', 'logit');
    end
    
    if isPositive && ~isBinary
        candidates{end+1} = struct('Dist', 'gamma', 'Link', 'log');
    end
    
    % ==========================================
    % 3. Candidate Models Fitting
    % ==========================================
    modelResults = struct('Dist', {}, 'AIC', {}, 'Model', {});
    validIdx = 0;
    
    for i = 1:length(candidates)
        try
            dist = candidates{i}.Dist;
            link = candidates{i}.Link;
            
            if strcmp(dist, 'normal')
                % Fit via ML ('Maximum Likelihood') is necessary to compare AIC of models 
                % with the same random component, though mathematically delicate when 
                % compared directly with GLMEs. Kept as a heuristic.
                mdl = fitlme(tbl, formula, 'FitMethod', 'ML');
            else
                % Fit with Laplace approximation for GLME (Robust R2025a standard)
                mdl = fitglme(tbl, formula, ...
                    'Distribution', dist, ...
                    'Link', link, ...
                    'FitMethod', 'Laplace');
            end
            
            validIdx = validIdx + 1;
            modelResults(validIdx).Dist  = dist;
            modelResults(validIdx).AIC   = mdl.ModelCriterion.AIC;
            modelResults(validIdx).Model = mdl;
            
            fprintf('Dist: %-8s | AIC: %.2f\n', dist, modelResults(validIdx).AIC);
            
        catch ME
            fprintf('⚠ Failed to fit %s: %s\n', dist, ME.message);
        end
    end
    
    % Model selection with the lowest Akaike Information Criterion
    [~, bestIdx] = min([modelResults.AIC]);
    bestModel = modelResults(bestIdx).Model;
    bestDist  = modelResults(bestIdx).Dist;
    
    % ==========================================
    % 4. Underlying Diagnostics (Overdispersion and Zero-Inflation)
    % ==========================================
    overdispersion = struct('phi', NaN, 'p', NaN);
    zeroInflation  = struct('flag', false, 'obs', NaN, 'exp', NaN);
    
    if isa(bestModel, 'GeneralizedLinearMixedModel') && strcmp(bestDist, 'poisson')
        
        % --- Overdispersion Analysis ---
        % Uses Pearson residuals according to fitglme specifications
        r = residuals(bestModel, 'ResidualType', 'Pearson');
        n = bestModel.NumObservations;
        p = length(fixedEffects(bestModel)); % Degrees of freedom based only on fixed effects
        
        chi2 = sum(r.^2, 'omitnan');
        df = n - p;
        
        phi = chi2 / df;
        pval = 1 - chi2cdf(chi2, df);
        
        overdispersion.phi = phi;
        overdispersion.p = pval;
        
        % --- Zero-Inflation Analysis ---
        % Approximate binomial criterion using marginal values ('Conditional', false)
        mu = predict(bestModel, 'Conditional', false);
        obsZero = mean(y_valid == 0);
        expZero = mean(exp(-mu(~isnan(mu)))); 
        
        % Approximate binomial standard error for tolerance
        seZero = sqrt(expZero * (1 - expZero) / n);
        
        zeroInflation.obs = obsZero;
        zeroInflation.exp = expZero;
        
        if obsZero > (expZero + 2*seZero)
            zeroInflation.flag = true;
        end
    end
    
    % ==========================================
    % 5. Output Formatting
    % ==========================================
    fprintf('\n--- Final Recommendation ---\nChosen Distribution: %s\n', upper(bestDist));
    
    results.BestModel = bestModel;
    if isa(bestModel, 'LinearMixedModel')
        results.Distribution = 'normal';
        results.modelType = 'lme';
    else
        results.Distribution = bestModel.Distribution;
        results.modelType = 'glme';
    end
    
    results.Diagnostics.Overdispersion = overdispersion;
    results.Diagnostics.ZeroInflation  = zeroInflation;
end