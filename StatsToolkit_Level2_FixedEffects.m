function Output = StatsToolkit_Level2_FixedEffects(mdl)
% STATSTOOLKIT_LEVEL2_FIXEDEFFECTS Computes effect sizes for fixed predictors.
%
% Syntax:
%   Output = StatsToolkit_Level2_FixedEffects(mdl)
%
% Inputs:
%   mdl - A fitted LinearMixedModel (LME) or GeneralizedLinearMixedModel (GLME).
%
% Outputs:
%   Output - Structure containing:
%       .AnovaTable        : Type III ANOVA table with effect size metrics.
%       .CoefficientsTable : Model coefficient estimates.
%       .ModelType         : String indicating 'lme' or 'glme'.
%
% Dependencies: Statistics and Machine Learning Toolbox (R2025a), StatsToolkit_Level1_GlobalMetrics

    %% =========================================================
    % 1. Extract ANOVA Table and Full Model Metadata
    %% =========================================================
    isGLME = isa(mdl, 'GeneralizedLinearMixedModel');
    AnovaTable = anova(mdl);
    
    if isGLME
        distName = lower(string(mdl.Distribution));
        linkName = lower(string(mdl.Link.Name));
    end
    
    try
        fullStats = StatsToolkit_Level1_GlobalMetrics(mdl);
        R2m_full = fullStats.R2_Marginal;
        canComputeR2 = true;
    catch ME
        fprintf('⚠ Failed to extract R2_Marginal from base model via Level1_GlobalMetrics: %s\n', ME.message);
        canComputeR2 = false;
        R2m_full = NaN;
    end
    
    %% =========================================================
    % 2. Vectorized Calculation (Partial Eta2 and Partial Omega2)
    %% =========================================================
    F_stat = AnovaTable.FStat;
    df1 = AnovaTable.DF1;
    df2 = AnovaTable.DF2;
    
    % Partial Eta-Squared (Universal for LME and GLME via FStat)
    eta2_p = (F_stat .* df1) ./ ((F_stat .* df1) + df2);
    
    if isGLME
        % Omega Squared remains formally undefined for GLME frameworks
        omega2_p = NaN(height(AnovaTable), 1);
    else
        % LME: Bias-corrected Partial Omega-Squared
        omega2_p = (df1 .* (F_stat - 1)) ./ (df1 .* (F_stat - 1) + df2 + 1);
        omega2_p(omega2_p < 0) = 0; % Cap negative estimates at 0
    end
    
    AnovaTable.PartialEtaSquared = eta2_p;
    AnovaTable.PartialOmegaSquared = omega2_p;
    
    %% =========================================================
    % 3. Iterative Calculation: Semi-Partial R-Squared (Algebraic Refit)
    %% =========================================================
    nTerms = height(AnovaTable);
    r2_sp = NaN(nTerms, 1);
    
    if canComputeR2
        tbl = mdl.Variables;
        responseVar = mdl.Formula.ResponseName;
        
        % Extract fixed terms directly from the ANOVA table
        allFixedTerms = string(AnovaTable.Term);
        allFixedTerms(allFixedTerms == "(Intercept)") = []; 
        
        % Isolate random components from the original formula (e.g., (1|ID))
        formulaStr = char(mdl.Formula);
        randomTerms = regexp(formulaStr, '\([^|]+\|[^)]+\)', 'match');
        
        if isempty(randomTerms)
            randStr = '';
        else
            randStr = [' + ' strjoin(randomTerms, ' + ')];
        end
        
        for i = 1:nTerms
            termName = char(AnovaTable.Term(i));
            
            if strcmp(termName, '(Intercept)')
                continue;
            end
            
            % Filter current term and all interactions containing it
            keepTerms = {};
            for t = 1:length(allFixedTerms)
                currentTerm = char(allFixedTerms(t));
                parts = split(currentTerm, ':');
                
                if ~any(strcmp(parts, termName))
                    keepTerms{end+1} = currentTerm;
                end
            end
            
            if isempty(keepTerms)
                fixedStr = '1';
            else
                fixedStr = strjoin(keepTerms, ' + ');
            end
            
            reducedFormula = sprintf('%s ~ %s%s', responseVar, fixedStr, randStr);
            
            try
                if isGLME
                    reducedMdl = fitglme(tbl, reducedFormula, ...
                        'Distribution', distName, ...
                        'Link', linkName, ...
                        'FitMethod', 'Laplace');
                else
                    % ML fit is mandatory here for valid fixed-effect comparison
                    reducedMdl = fitlme(tbl, reducedFormula, 'FitMethod', 'ML');
                end
                
                redStats = StatsToolkit_Level1_GlobalMetrics(reducedMdl);
                r2_delta = R2m_full - redStats.R2_Marginal;
                
                if r2_delta < 0
                    r2_delta = 0;
                end
                
                r2_sp(i) = r2_delta;
                
            catch ME
                fprintf('⚠ Reduced model (Term: %s) skipped. Convergence failure: %s\n', termName, ME.message);
                r2_sp(i) = NaN;
            end
        end
    end
    AnovaTable.SemiPartial_R2 = r2_sp;
    
    %% =========================================================
    % 4. Heuristic Magnitude Classification (Cohen, 1988)
    %% =========================================================
    Magnitude = strings(nTerms, 1);
    
    if isGLME
        metric = eta2_p; 
    else
        metric = omega2_p; 
    end
    
    Magnitude(metric < 0.01) = "Negligible";
    Magnitude(metric >= 0.01 & metric < 0.06) = "Small";
    Magnitude(metric >= 0.06 & metric < 0.14) = "Medium";
    Magnitude(metric >= 0.14) = "Large";
    Magnitude(strcmp(AnovaTable.Term, '(Intercept)')) = "-";
    AnovaTable.Magnitude = Magnitude;
    
    %% =========================================================
    % 5. Output Structuring
    %% =========================================================
    Output = struct();
    Output.AnovaTable = AnovaTable;
    Output.CoefficientsTable = mdl.Coefficients;
    
    if isGLME
        Output.ModelType = 'glme';
    else
        Output.ModelType = 'lme';
    end
    
    disp('===========================================================');
    disp('            FIXED EFFECTS AND EFFECT SIZES                 ');
    disp('===========================================================');
    disp('--- ANOVA Table (Type III Marginal Tests) ---');
    disp(Output.AnovaTable);
    disp('--- Coefficient Estimates Table ---');
    disp(Output.CoefficientsTable);
end