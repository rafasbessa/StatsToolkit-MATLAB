function Output = StatsToolkit_Level1_SelectRandomSlopes(tbl, distResults, savepath, user_formulas, transformed_data, userNslopes)
% STATSTOOLKIT_LEVEL1_SELECTRANDOMSLOPES Evaluates and selects the optimal random slope structure.
%
% Refined Version: Includes Restricted Observation-Level Random Effects (OLRE) 
% and isolated user-defined formulas.
%
% Syntax:
%   Output = StatsToolkit_Level1_SelectRandomSlopes(tbl, distResults, savepath, user_formulas, transformed_data, userNslopes)
%
% Inputs:
%   tbl              - Data table.
%   distResults      - Structure from StatsToolkit_Level1_SelectDistribution.
%   savepath         - Directory path to save the generated QQ plot.
%   user_formulas    - (Optional) Cell array of custom formula strings.
%   transformed_data - (Optional) Table of pre-transformed variables.
%   userNslopes      - (Optional) Numeric array indicating complexity of user formulas.
%
% Outputs:
%   Output - Structure containing:
%       .ComparisonTable         : Table with metrics for all fitted models.
%       .ModelObjects            : Cell array of fitted model objects.
%       .SuggestedModelIndex     : Index of the recommended model.
%       .SuggestedSlopeStructure : String describing the selected random structure.
%
% Dependencies: Statistics and Machine Learning Toolbox (R2025a)

    if nargin < 4, user_formulas = []; transformed_data = table; userNslopes = []; end

    %% =========================================================
    % 1. Extract Model Information
    %% =========================================================
    modelType = lower(distResults.modelType);
    baseModel = distResults.BestModel;
    
    varnames = tbl.Properties.VariableNames;
    response = varnames{end};
    groupVar = varnames{1};
    fixedVars = varnames(2:end-1);
    
    distribution = [];
    link = [];
    if strcmp(modelType, 'glme')
        distribution = lower(string(baseModel.Distribution));
        link = lower(string(baseModel.Link.Name));
    end

    %% =========================================================
    % Observation-Level Random Effect (OLRE) Index
    %% =========================================================
    tbl.OLRE_ID = (1:height(tbl))';

    %% =========================================================
    % 2. Detect Candidate Slope Variables
    %% =========================================================
    basePredictors = {};
    candidateSlopes = {};
    
    G = findgroups(tbl.(groupVar));
    for i = 1:length(fixedVars)
        varName = fixedVars{i};
        varData = tbl.(varName);
        
        % Check if variable has within-group variance
        varWithin = splitapply(@(x) numel(unique(x)) > 1, varData, G);
        if ~any(varWithin), continue; end
        
        basePredictors{end+1} = varName;
        
        if ~iscategorical(varData)
            candidateSlopes{end+1} = varName;
        else
            candidateSlopes{end+1} = varName;
            contName = [varName '_cont'];
            
            % Create continuous version of categorical variable if it doesn't exist
            if ~ismember(contName, tbl.Properties.VariableNames)
                tbl.(contName) = double(categorical(varData));
            end
            candidateSlopes{end+1} = contName;
        end
    end

    %% =========================================================
    % Interaction Term
    %% =========================================================
    if length(basePredictors) > 1
        interTerm = strjoin(basePredictors, ':');
        candidateSlopes{end+1} = interTerm;
    end

    %% =========================================================
    % Combination Space (Corrected to avoid redundancy)
    %% =========================================================
    k = length(candidateSlopes);
    slopeComb = {{}};
    
    if k <= 4
        for mask = 1:(2^k - 1)
            idx = logical(bitget(mask, 1:k));
            comb = candidateSlopes(idx);
            valid = true;
            
            % Rule 1: Interaction Validation
            for j = 1:length(comb)
                if contains(comb{j}, ':')
                    parts = split(comb{j}, ':');
                    if ~all(ismember(parts, comb)), valid = false; end
                end
            end
            
            % Rule 2: Mutual exclusion (Categorical vs Continuous) - Essential for Rank
            for v = 1:length(fixedVars)
                orig = fixedVars{v};
                cont = [orig '_cont'];
                if ismember(orig, comb) && ismember(cont, comb)
                    valid = false; break; 
                end
            end
            
            if valid, slopeComb{end+1} = comb; end
        end
    else
        slopeComb = {{}};
        
        % Add individual slopes
        for i = 1:k, slopeComb{end+1} = {candidateSlopes{i}}; end
        
        % Add Maximum Categorical model (No redundancy)
        maxCat = candidateSlopes(~contains(candidateSlopes, '_cont'));
        slopeComb{end+1} = maxCat;
        
        % Add Maximum Continuous model (Replacing originals with continuous when available)
        maxCont = {};
        for i = 1:length(candidateSlopes)
            curr = candidateSlopes{i};
            if contains(curr, '_cont')
                maxCont{end+1} = curr;
            elseif ~ismember([curr '_cont'], candidateSlopes)
                maxCont{end+1} = curr;
            end
        end
        if ~isempty(maxCont) && ~isequal(sort(maxCont), sort(maxCat))
            slopeComb{end+1} = maxCont;
        end
    end

    %% =========================================================
    % Build Formulas
    %% =========================================================
    fixedFormula = strjoin(fixedVars, ' * ');
    nStandardModels = length(slopeComb);
    
    formulas = cell(nStandardModels, 1);
    labels = cell(nStandardModels, 1);
    complexity = zeros(nStandardModels, 1);
    
    added_cont_baselines = {}; 
    fixedFormulaCont_ref = ''; % Stores continuous version for subsequent OLRE
    
    for i = 1:nStandardModels
        slopes = slopeComb{i};
        
        if isempty(slopes)
            randPart = ['(1|' groupVar ')'];
            labels{i} = 'Intercept only';
            complexity(i) = 0;
            formulas{i} = [response ' ~ ' fixedFormula ' + ' randPart];
            
        elseif any(contains(slopes, '_cont'))
            slopeStr = strjoin(slopes, ' + ');
            randPart = ['(1 + ' slopeStr '|' groupVar ')'];
            labels{i} = slopeStr;
            complexity(i) = length(slopes);
            
            fixedVarsCont = fixedVars;
            for sIdx = 1:length(slopes)
                if contains(slopes{sIdx}, '_cont')
                    origVar = strrep(slopes{sIdx}, '_cont', '');
                    idxMatch = strcmp(fixedVars, origVar);
                    if any(idxMatch)
                        fixedVarsCont{idxMatch} = slopes{sIdx};
                    end
                end
            end
            
            fixedFormulaCont = strjoin(fixedVarsCont, ' * ');
            fixedFormulaCont_ref = fixedFormulaCont; % Reference for restricted OLRE
            formulas{i} = [response ' ~ ' fixedFormulaCont ' + ' randPart];
            
            if ~ismember(fixedFormulaCont, added_cont_baselines)
                formulas = [formulas; {[response ' ~ ' fixedFormulaCont ' + (1|' groupVar ')']}];
                labels = [labels; {'Intercept only (cont fixed)'}];
                complexity = [complexity; 0];
                added_cont_baselines{end+1} = fixedFormulaCont;
            end
            
        else
            slopeStr = strjoin(slopes, ' + ');
            randPart = ['(1 + ' slopeStr '|' groupVar ')'];
            labels{i} = slopeStr;
            complexity(i) = length(slopes);
            formulas{i} = [response ' ~ ' fixedFormula ' + ' randPart];
        end
    end

    %% =========================================================
    % Poisson OLRE Addition (Restricted to Intercept-only Candidates)
    %% =========================================================
    baseStats = StatsToolkit_Level1_GlobalMetrics(baseModel);
    if strcmp(modelType, 'glme') && strcmp(distribution, 'poisson') && ...
            isfield(baseStats, 'Phi') && baseStats.Phi > 1.1
        
        fprintf('\n--- Poisson Overdispersion detected (Phi=%.2f). Adding Restricted OLRE models.\n', baseStats.Phi);
        
        % OLRE strictly on categorical intercept
        formulas{end+1} = [response ' ~ ' fixedFormula ' + (1|' groupVar ') + (1|OLRE_ID)'];
        labels{end+1} = 'OLRE (Categorical Intercept)';
        complexity(end+1) = 1;
        
        % OLRE strictly on continuous intercept (if continuous version exists)
        if ~isempty(fixedFormulaCont_ref)
            formulas{end+1} = [response ' ~ ' fixedFormulaCont_ref ' + (1|' groupVar ') + (1|OLRE_ID)'];
            labels{end+1} = 'OLRE (Continuous Intercept)';
            complexity(end+1) = 1;
        end
    end

    %% =========================================================
    % Add User Formulas and Transformed Variables
    %% =========================================================
    if iscell(user_formulas) && ~isempty(user_formulas)
        user_labels = cell(length(user_formulas), 1);
        for i = 1:length(user_formulas)
            s = strfind(user_formulas{i}, '(');
            if ~isempty(s)
                user_labels{i,1} = user_formulas{i}(s(1):end);
            else
                user_labels{i,1} = 'User Formula';
            end
        end
        complexity = [complexity; userNslopes(:)];
        labels = [labels; user_labels];
        formulas = [formulas; user_formulas(:)];
    end
    tbl = [tbl, transformed_data];

    %% =========================================================
    % Model Fitting Pipeline (Full Transparency)
    %% =========================================================
    nTotalModels = length(formulas);
    Results = [];
    ModelObjects = cell(nTotalModels, 1);
    disp(['Total Models: ' num2str(nTotalModels)])
    
    for i = 1:nTotalModels
        disp(['Model ' num2str(i) ': ' formulas{i}])
        try
            if strcmp(modelType, 'lme')
                mdl = fitlme(tbl, formulas{i}, 'FitMethod', 'ML');
            else
                mdl = fitglme(tbl, formulas{i}, ...
                    'Distribution', distribution, ...
                    'Link', link, ...
                    'FitMethod', 'Laplace');
            end
            
            hasEstimates = ~any(isnan(mdl.Coefficients.Estimate));
            fitOK = isfinite(mdl.ModelCriterion.LogLikelihood) && hasEstimates;
            
            singularFlag = false;
            try
                [~, ~, stats_re] = covarianceParameters(mdl);
                for s = 1:numel(stats_re)
                    est = stats_re{s}.Estimate;
                    if any(~isfinite(est)) || any(abs(est) < 1e-8)
                        singularFlag = true; break;
                    end
                end
            catch
                singularFlag = true;
            end
            
            p = numel(mdl.Coefficients.Estimate);
            sampleOK = mdl.NumObservations > (p + 1);
            stats = StatsToolkit_Level1_GlobalMetrics(mdl);
            ModelObjects{i} = mdl;
            
            resEntry = struct( ...
                'ModelIndex', i, ...
                'Formula', {formulas{i}}, ...
                'SlopeStructure', {labels{i}}, ...
                'Complexity', complexity(i), ...
                'fitOK', fitOK, ...
                'isSingular', singularFlag, ...
                'sampleOK', sampleOK, ...
                'AICc', stats.AICc, ...
                'AIC', mdl.ModelCriterion.AIC, ...
                'BIC', mdl.ModelCriterion.BIC, ...
                'LogLikelihood', mdl.ModelCriterion.LogLikelihood, ...
                'R2_Marginal', stats.R2_Marginal, ...
                'R2_Conditional', stats.R2_Conditional, ...
                'ICC', stats.ICC, ...
                'f2', stats.f2, ...
                'Phi', stats.Phi );
            
            if isempty(Results)
                Results = resEntry; 
            else 
                Results(end+1) = resEntry; 
            end
            
        catch
            continue
        end
    end
    
    if isempty(Results), error('No slope models could be fitted.'); end

    %% =========================================================
    % Table & Diagnostics Plot
    %% =========================================================
    CompTable = struct2table(Results);
    CompTable.DeltaAICc = CompTable.AICc - min(CompTable.AICc);
    
    h = figure('Name', 'Residual QQ Diagnostics');
    tiledlayout('flow');
    
    for i = 1:length(ModelObjects)
        mdl = ModelObjects{i};
        if isempty(mdl), continue; end
        
        rowIdx = find(CompTable.ModelIndex == i);
        if isempty(rowIdx), continue; end
        
        nexttile
        try
            r = residuals(mdl, 'ResidualType', 'Pearson');
            qqplot(r)
            
            statusTag = '';
            if ~CompTable.fitOK(rowIdx), statusTag = [statusTag ' !NoConv']; end
            if CompTable.isSingular(rowIdx), statusTag = [statusTag ' !Singular']; end
            
            title(sprintf('M%d: %s%s', i, labels{i}, statusTag), ...
                'Interpreter', 'none', 'FontSize', 12)
            set(gca, 'LineWidth', 2, 'FontSize', 12)
        catch
            text(0.5, 0.5, 'Plot Error', 'HorizontalAlignment', 'center');
        end
    end

    %% =========================================================
    % Model Suggestion
    %% =========================================================
    validIdx = CompTable.fitOK & ~CompTable.isSingular & CompTable.sampleOK;
    
    if any(validIdx)
        candidates = CompTable(validIdx, :);
        minAICc = min(candidates.AICc);
        bestCandidates = candidates(candidates.AICc == minAICc, :);
        [~, bIdx] = min(bestCandidates.Complexity);
        bestRow = bestCandidates(bIdx, :);
    else
        [~, bIdx] = min(CompTable.AICc);
        bestRow = CompTable(bIdx, :);
    end

    Output = struct;
    Output.ComparisonTable = CompTable;
    Output.ModelObjects = ModelObjects;
    Output.SuggestedModelIndex = bestRow.ModelIndex;
    Output.SuggestedSlopeStructure = bestRow.SlopeStructure;

    disp('--- Final Model Comparison ---')
    disp(CompTable)
    fprintf('Suggested Random Structure: %s (%d)\n', char(bestRow.SlopeStructure), Output.SuggestedModelIndex)
    
    exportgraphics(h, [savepath response '_QQplot.png'], 'Resolution', 300);
end