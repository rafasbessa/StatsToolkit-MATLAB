function ResultsTable = StatsToolkit_Level4_CompositionalInverse(modelsCell, predictionGrid, method, options)
% STATSTOOLKIT_LEVEL4_COMPOSITIONALINVERSE Inverse transformations for compositional data.
%
% Extracts predictions from multiple LME/GLME models fitted to transformed 
% coordinates and back-transforms them to the original simplex (proportions).
%
% Supported Methods:
%   'ilr' - Inverse Isometric Log-Ratio (Expects D-1 models)
%   'clr' - Inverse Centered Log-Ratio (Expects D models)
%   'alr' - Inverse Additive Log-Ratio (Expects D-1 models)
%
% Syntax:
%   ResultsTable = StatsToolkit_Level4_CompositionalInverse(modelsCell, predictionGrid, method, Name, Value)
%
% Inputs:
%   modelsCell     - Cell array containing the fitted mixed model objects.
%   predictionGrid - Table containing the fixed-effect conditions to predict on.
%   method         - String: 'ilr', 'clr', or 'alr'.
%   options:
%       'ReferenceColumn' - (ALR only) The reference part index used in the 
%                           forward transformation. Defaults to the last part (D).
%
% Outputs:
%   ResultsTable   - Table containing the fixed conditions and the back-transformed proportions.
%
% Dependencies: Statistics and Machine Learning Toolbox (R2025a)

    arguments
        modelsCell cell {mustBeNonempty}
        predictionGrid table {mustBeNonempty}
        method string {mustBeMember(method, {'ilr', 'clr', 'alr'})}
        options.ReferenceColumn (1,1) double = 0
    end

    numModels = length(modelsCell);
    numObs = height(predictionGrid);
    
    %% =========================================================
    % 1. Add Placeholders for Random Variables
    %% =========================================================
    % MATLAB's predict function requires all variables (including grouping vars)
    % to be present in the dataset, even if Conditional = false.
    groupingVars = string(modelsCell{1}.Formula.GroupingVariableNames);
    originalData = modelsCell{1}.Variables;
    
    for g = 1:numel(groupingVars)
        varName = char(groupingVars(g));
        % If the grouping variable is not in the grid, add a dummy column
        if ~ismember(varName, predictionGrid.Properties.VariableNames)
            predictionGrid.(varName) = repmat(originalData.(varName)(1), numObs, 1);
        end
    end
    
    %% =========================================================
    % 2. Extract Marginal Predictions
    %% =========================================================
    % Matrix to store the marginal predictions from each model
    predMatrix = zeros(numObs, numModels);
    
    for i = 1:numModels
        mdl = modelsCell{i};
        % 'Conditional', false extracts the population-level (fixed) effects
        predMatrix(:, i) = predict(mdl, predictionGrid, 'Conditional', false);
    end

    %% =========================================================
    % 3. Apply Inverse Transformations
    %% =========================================================
    switch method
        
        case 'clr'
            % For CLR, the number of models equals the number of parts (D)
            D = numModels;
            
            % Inverse CLR is simply the softmax/closure operation
            exp_vals = exp(predMatrix);
            proportions = exp_vals ./ sum(exp_vals, 2);
            
        case 'ilr'
            % For ILR, the number of models is D-1
            D = numModels + 1;
            
            % Dynamically reconstruct the identical Helmert orthonormal basis (V) 
            % used in StatsToolkit_Level0_Transform
            V = zeros(D, D-1);
            for j = 1:D-1
                h = 1 / sqrt(j * (j + 1));
                V(1:j, j) = h;
                V(j+1, j) = -j * h;
            end
            
            % 1. Project from ILR back to CLR space
            clr_matrix = predMatrix * V';
            
            % 2. Apply exponential and closure (softmax)
            exp_clr = exp(clr_matrix);
            proportions = exp_clr ./ sum(exp_clr, 2);
            
        case 'alr'
            % For ALR, the number of models is D-1
            D = numModels + 1;
            
            refCol = options.ReferenceColumn;
            if refCol == 0 || refCol > D
                refCol = D; % Default: reference was the last column
            end
            
            % The ALR inverse uses 1 for the reference column (since ln(X/X) = 0, exp(0) = 1)
            % and exp(predicted_ratio) for the other columns.
            exp_alr = exp(predMatrix);
            
            full_matrix = zeros(numObs, D);
            full_matrix(:, refCol) = 1; % Set reference part to 1
            
            % Map the remaining parts back to their original column indices
            otherCols = setdiff(1:D, refCol);
            full_matrix(:, otherCols) = exp_alr;
            
            % Apply closure operation to return to the simplex
            proportions = full_matrix ./ sum(full_matrix, 2);
            
    end

    %% =========================================================
    % 4. Output Formatting
    %% =========================================================
    % Create dynamic column names based on the number of parts
    propNames = compose('Prop_Part%d', 1:D);
    propTable = array2table(proportions, 'VariableNames', propNames);
    
    % Remove the dummy random variables before returning the table
    for g = 1:numel(groupingVars)
        varName = char(groupingVars(g));
        predictionGrid.(varName) = [];
    end
    
    ResultsTable = [predictionGrid, propTable];
    
    fprintf('✅ Success: Inverse %s transformation applied.\n', upper(method));
    fprintf('ℹ %d models mapped to %d compositional parts for %d conditions.\n', numModels, D, numObs);
end