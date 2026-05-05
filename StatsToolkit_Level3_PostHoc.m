function PostHocTable = StatsToolkit_Level3_PostHoc(mdl, Level2_Output, pAdjusted, options)
% STATSTOOLKIT_LEVEL3_POSTHOC Unifies Pairwise Comparisons and Simple Slopes.
% Supports N-Way interactions and marginalization over reference grids.
%
% Syntax:
%   PostHocTable = StatsToolkit_Level3_PostHoc(mdl, Level2_Output, pAdjusted, Name, Value)
%
% Inputs:
%   mdl           - Fitted LinearMixedModel or GeneralizedLinearMixedModel.
%   Level2_Output - Structure returned by StatsToolkit_Level2_FixedEffects.
%   pAdjusted     - (Optional) Custom vector of p-values to override ANOVA table.
%   options:
%       'alpha'     - Significance level (default: 0.05).
%       'adjMethod' - Multiple comparisons adjustment: 'FDR', 'HOLM', 'SIDAK', 'TUKEY' (default: 'FDR').
%
% Outputs:
%   PostHocTable  - Table containing contrast estimates, standard errors, and adjusted p-values.
%
% Dependencies: Statistics and Machine Learning Toolbox (R2025a)

    arguments
        mdl {mustBeA(mdl, ["LinearMixedModel", "GeneralizedLinearMixedModel"])}
        Level2_Output struct {mustBeNonempty}
        pAdjusted (:,1) double = []
        options.alpha (1,1) double = 0.05
        options.adjMethod (1,1) string {mustBeMember(options.adjMethod, ["FDR", "HOLM", "SIDAK", "TUKEY"])} = 'FDR'
    end
    
    alpha = options.alpha;
    adjMethod = options.adjMethod;
    AnovaTable = Level2_Output.AnovaTable;
    
    isGLME = isa(mdl, 'GeneralizedLinearMixedModel');
    isLME  = isa(mdl, 'LinearMixedModel');
    
    tbl = mdl.Variables;
    beta = mdl.Coefficients.Estimate;
    Sigma = mdl.CoefficientCovariance;
    coefNames = mdl.CoefficientNames;
    
    %% =========================================================
    % 1. Residual Variance for Cohen's d (Latent Scale if GLME)
    %% =========================================================
    try
        if isLME
            [~, sig2] = covarianceParameters(mdl);
            var_resid = double(sig2);
            df = mdl.DFE;
        elseif isGLME
            [~, sig2] = covarianceParameters(mdl);
            distName = lower(mdl.Distribution.Name);
            if strcmp(distName, 'poisson')
                mu_bar = mean(predict(mdl, 'Conditional', false), 'omitnan');
                var_resid = log(1 + 1/mu_bar); 
            else
                var_resid = double(sig2);
            end
            df = inf;
        end
    catch
        var_resid = 1; 
        df = inf;
    end
    
    %% =========================================================
    % 2. Identification of Variables and Significant Terms
    %% =========================================================
    if ~isempty(pAdjusted)
        sigIdx = pAdjusted < alpha;
    else
        sigIdx = AnovaTable.pValue < alpha;
    end
        
    sigTerms = AnovaTable.Term(sigIdx);
    predictors = mdl.Formula.PredictorNames;
    
    contVars = {}; 
    contMeans = [];
    isCatMap = containers.Map();
    
    for i = 1:length(predictors)
        v = predictors{i};
        if iscategorical(tbl.(v))
            isCatMap(v) = true;
        else
            isCatMap(v) = false;
            contVars{end+1} = v;
            contMeans(end+1) = mean(tbl.(v), 'omitnan');
        end
    end
    results = [];
    
    %% =========================================================
    % 3. N-Way Processing Loop (Pairwise vs Simple Slope)
    %% =========================================================
    for i = 1:length(sigTerms)
        termName = char(sigTerms(i));
        if strcmp(termName, '(Intercept)'), continue; end
        
        parts = split(termName, ':');
        hasCont = false;
        contInTerm = '';
        catInTerm = {};
        
        for p = 1:length(parts)
            if ~isCatMap(parts{p})
                hasCont = true;
                contInTerm = parts{p};
            else
                catInTerm{end+1} = parts{p};
            end
        end
        
        % Identify external categorical variables for global marginalization
        allCatVars = predictors(cell2mat(isCatMap.values));
        nuisanceVars = setdiff(allCatVars, parts);
        nuisanceGrid = buildCategoricalGrid(tbl, nuisanceVars);
        
        if ~hasCont
            %% ---------------------------------------------------------
            % CASE 1: PAIRWISE COMPARISONS (Purely Categorical)
            %% ---------------------------------------------------------
            for tgtIdx = 1:length(parts)
                targetFactor = parts{tgtIdx};
                condFactors = parts([1:tgtIdx-1, tgtIdx+1:end]);
                
                targetLevels = categories(tbl.(targetFactor));
                pairs = nchoosek(1:length(targetLevels), 2);
                
                condGrid = buildCategoricalGrid(tbl, condFactors);
                nConds = max(1, height(condGrid));
                
                for c = 1:nConds
                    % Index protection for main effects
                    if isempty(condFactors)
                        condRow = table();
                    else
                        condRow = condGrid(c, :);
                    end
                    condStr = buildCondStr(condFactors, condRow);
                    
                    for j = 1:size(pairs, 1)
                        L1 = targetLevels{pairs(j, 1)}; 
                        L2 = targetLevels{pairs(j, 2)};
                        
                        tblA = assembleReferenceGrid(tbl, targetFactor, L1, condRow, nuisanceGrid, contVars, contMeans);
                        tblB = assembleReferenceGrid(tbl, targetFactor, L2, condRow, nuisanceGrid, contVars, contMeans);
                        
                        [E, SE, t_stat, p_val, d] = computeMarginalContrast(tblA, tblB, beta, Sigma, df, var_resid, coefNames, contVars);
                        results = appendResult(results, termName, targetFactor, condStr, sprintf('%s vs %s', L1, L2), E, SE, t_stat, p_val, d);
                    end
                end
                if isempty(condFactors), break; end
            end
        else
            %% ---------------------------------------------------------
            % CASE 2: SIMPLE SLOPES (Contains Continuous Covariate)
            %% ---------------------------------------------------------
            condGrid = buildCategoricalGrid(tbl, catInTerm);
            nConds = max(1, height(condGrid));
            
            for c = 1:nConds
                if isempty(catInTerm)
                    condRow = table();
                else
                    condRow = condGrid(c, :);
                end
                condStr = buildCondStr(catInTerm, condRow);
                
                % Slope calculated via numerical derivative (Delta X = 1)
                tblX1 = assembleReferenceGrid(tbl, contInTerm, 1, condRow, nuisanceGrid, contVars, contMeans);
                tblX0 = assembleReferenceGrid(tbl, contInTerm, 0, condRow, nuisanceGrid, contVars, contMeans);
                
                [E, SE, t_stat, p_val, d] = computeMarginalContrast(tblX1, tblX0, beta, Sigma, df, var_resid, coefNames, contVars);
                results = appendResult(results, termName, contInTerm, condStr, 'Simple Slope', E, SE, t_stat, p_val, d);
            end
        end
    end
    
    %% =========================================================
    % 4. Final Table and Multiple Comparisons Adjustment
    %% =========================================================
    if isempty(results)
        disp('No applicable post-hoc or significant term detected.');
        PostHocTable = table(); 
        return;
    end
    
    PostHocTable = struct2table(results);
    m = height(PostHocTable);
    p_adj = zeros(m, 1);
    [~, idx] = sort(PostHocTable.p_uncorr);
    
    switch upper(adjMethod)
        case 'FDR'
            for k = 1:m, p_adj(idx(k)) = min(1, PostHocTable.p_uncorr(idx(k)) * (m/k)); end
            for k = m-1:-1:1, p_adj(idx(k)) = min(p_adj(idx(k)), p_adj(idx(k+1))); end
        case 'HOLM'
            for k = 1:m, p_adj(idx(k)) = min(1, PostHocTable.p_uncorr(idx(k)) * (m - k + 1)); end
            for k = 2:m, p_adj(idx(k)) = max(p_adj(idx(k)), p_adj(idx(k-1))); end
        case {'SIDAK', 'TUKEY'}
            p_adj = 1 - (1 - PostHocTable.p_uncorr).^m;
        otherwise
            error('Invalid method. Choose ''FDR'', ''HOLM'', ''SIDAK'', or ''TUKEY''.');
    end
    
    PostHocTable.adjMethod = repmat(adjMethod, height(PostHocTable), 1);
    PostHocTable.p_adj = p_adj;
    PostHocTable.Significant = p_adj < alpha;
    
    disp('===========================================================')
    fprintf('       POST-HOC ANALYSIS: PAIRWISE & SLOPES (%s)\n', upper(adjMethod))
    disp('===========================================================')
    disp(PostHocTable)
end

%% =========================================================
% Local Auxiliary Functions
%% =========================================================
function gridTbl = buildCategoricalGrid(tbl, varNames)
    if isempty(varNames), gridTbl = table(); return; end
    levelsList = cell(1, length(varNames));
    idxList = cell(1, length(varNames));
    for i = 1:length(varNames)
        levelsList{i} = categories(categorical(tbl.(varNames{i})));
        idxList{i} = 1:length(levelsList{i});
    end
    gridArray = cell(1, length(varNames));
    [gridArray{:}] = ndgrid(idxList{:});
    gridTbl = table();
    for i = 1:length(varNames)
        levs = levelsList{i}; 
        idx = gridArray{i}(:);
        gridTbl.(varNames{i}) = categorical(levs(idx), levelsList{i});
    end
end

function fullGridTbl = assembleReferenceGrid(baseTbl, targetVar, targetLevel, condRow, nuisanceGrid, contVars, contMeans)
    nRows = max(1, height(nuisanceGrid));
    fullGridTbl = repmat(baseTbl(1, :), nRows, 1);
    for v = 1:length(contVars)
        fullGridTbl.(contVars{v})(:) = contMeans(v);
    end
    if ~isempty(nuisanceGrid)
        nVars = nuisanceGrid.Properties.VariableNames;
        for v = 1:length(nVars), fullGridTbl.(nVars{v}) = nuisanceGrid.(nVars{v}); end
    end
    if ~isempty(condRow)
        cVars = condRow.Properties.VariableNames;
        for v = 1:length(cVars), fullGridTbl.(cVars{v})(:) = condRow.(cVars{v}); end
    end
    
    % Type Correction: Categorical vs Continuous
    if iscategorical(baseTbl.(targetVar))
        fullGridTbl.(targetVar)(:) = categorical({targetLevel}, categories(baseTbl.(targetVar)));
    else
        fullGridTbl.(targetVar)(:) = targetLevel;
    end
end

function X_mat = buildDesignMatrixManual(tblGrid, coefNames, contVars)
    nRows = height(tblGrid); 
    nCoef = length(coefNames);
    X_mat = zeros(nRows, nCoef);
    varNames = tblGrid.Properties.VariableNames;
    isCat = false(1, length(varNames));
    
    for v = 1:length(varNames)
        isCat(v) = iscategorical(tblGrid.(varNames{v})); 
    end
    
    for r = 1:nRows
        rowTbl = tblGrid(r, :);
        for i = 1:nCoef
            cName = char(coefNames{i});
            if strcmp(cName, '(Intercept)')
                X_mat(r, i) = 1; 
                continue; 
            end
            parts = split(cName, ':'); 
            val = 1;
            for p = 1:length(parts)
                part = char(parts{p});
                if any(strcmp(contVars, part))
                    val = val * double(rowTbl.(part)); 
                    continue;
                end
                matchedCat = false;
                for v = 1:length(varNames)
                    if isCat(v)
                        prefix = [varNames{v} '_'];
                        if startsWith(part, prefix)
                            levelName = extractAfter(part, prefix);
                            if strcmp(char(rowTbl.(varNames{v})), levelName)
                                val = val * 1; 
                            else 
                                val = val * 0; 
                            end
                            matchedCat = true; 
                            break;
                        end
                    end
                end
                if ~matchedCat, val = 0; end
            end
            X_mat(r, i) = val;
        end
    end
end

function [E, SE, t_stat, p_val, d] = computeMarginalContrast(tblA, tblB, beta, Sigma, df, var_resid, coefNames, contVars)
    XA_full = buildDesignMatrixManual(tblA, coefNames, contVars);
    XB_full = buildDesignMatrixManual(tblB, coefNames, contVars);
    XA_marg = mean(XA_full, 1); 
    XB_marg = mean(XB_full, 1);
    
    L = XA_marg - XB_marg;
    E = L * beta; 
    SE = sqrt(L * Sigma * L');
    t_stat = E / SE;
    
    if isinf(df)
        p_val = 2 * normcdf(-abs(t_stat)); 
    else 
        p_val = 2 * tcdf(-abs(t_stat), df); 
    end
    
    d = E / sqrt(var_resid);
end

function s = buildCondStr(factors, row)
    if isempty(factors)
        s = '-'; 
        return; 
    end
    strs = cell(1, length(factors));
    for i = 1:length(factors)
        strs{i} = sprintf('%s=%s', factors{i}, char(row{1, factors{i}})); 
    end
    s = strjoin(strs, ', ');
end

function res = appendResult(res, term, target, cond, comp, E, SE, t, p, d)
    entry.Term = term; 
    entry.Target = target; 
    entry.Condition = cond;
    entry.Comparison = comp; 
    entry.Estimate = E; 
    entry.SE = SE;
    entry.tStat = t; 
    entry.p_uncorr = p; 
    entry.Cohen_d = d;
    
    if isempty(res)
        res = entry; 
    else 
        res(end+1) = entry; 
    end
end