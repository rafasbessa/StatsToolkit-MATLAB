function dataTransformed = StatsToolkit_Level0_Transform(data, method, options)
% STATSTOOLKIT_LEVEL0_TRANSFORM Data preprocessing and mathematical transformations.
%
% Supported Methods:
%   'clr'    - Centered Log-Ratio (Compositional data)
%   'ilr'    - Isometric Log-Ratio (Orthonormal projection for regression)
%   'alr'    - Additive Log-Ratio (Proportions relative to a reference part)
%   'logit'  - Logit Transformation (Proportions bounded 0 to 1)
%   'zscore' - Standardization (Mean 0, SD 1)
%   'log'    - Natural logarithm (with offset for zeros)
%
% Syntax:
%   dataTransformed = StatsToolkit_Level0_Transform(data, method, Name, Value)
%
% Inputs:
%   data    - Numeric matrix of data to be transformed.
%   method  - String specifying the transformation method.
%   options:
%       'Offset'          - Value added to handle zeros (default: 1e-6).
%       'ReferenceColumn' - Reference column index for ALR (default: 0, uses last column).
%
% Outputs:
%   dataTransformed - Matrix containing the transformed data.
%
% Dependencies: MATLAB Base (R2025a)

    arguments
        data double {mustBeNonempty}
        method string {mustBeMember(method, {'clr', 'ilr', 'alr', 'logit', 'zscore', 'log'})}
        options.Offset (1,1) double = 1e-6
        options.ReferenceColumn (1,1) double = 0
    end
    
    [nRows, nCols] = size(data);
    
    %% =========================================================
    % 1. Compositional Data (Simplex) Detection
    %% =========================================================
    if ismember(method, {'clr', 'ilr', 'alr'})
        
        rowSums = sum(data, 2, 'omitnan');
        
        % Numerical tolerance
        tol = 1e-6;
        
        if any(abs(rowSums - 1) > tol)
            % Assumes compositional counts → normalizes to relative proportions
            fprintf('ℹ Data is not in the simplex. Normalizing rows to proportions.\n')
            
            rowSums(rowSums == 0) = NaN; % Prevents division by zero
            data = data ./ rowSums;
        end
        
    end
    
    %% =========================================================
    % 2. Zero Handling for Logarithmic Methods
    %% =========================================================
    if ismember(method, {'clr', 'ilr', 'alr', 'log', 'logit'})
        data(data <= 0) = options.Offset;
    end
    
    %% =========================================================
    % 3. Transformations
    %% =========================================================
    switch method
        
        case 'clr'
            % Centered Log-Ratio: ln(xi / g(x))
            geoMean = exp(mean(log(data), 2, 'omitnan'));
            dataTransformed = log(data ./ geoMean);
            
        case 'ilr'
            % Isometric Log-Ratio: Projects D parts into D-1 coordinates.
            % Utilizes Helmert orthonormal basis.
            
            D = nCols;
            V = zeros(D, D-1);
            
            for j = 1:D-1
                h = 1 / sqrt(j * (j + 1));
                V(1:j, j) = h;
                V(j+1, j) = -j * h;
            end
            
            clrData = log(data ./ exp(mean(log(data), 2, 'omitnan')));
            dataTransformed = clrData * V;
            
        case 'alr'
            % Additive Log-Ratio: ln(xi / x_ref)
            refCol = options.ReferenceColumn;
            if refCol == 0 || refCol > nCols
                refCol = nCols; % Default: uses the last column as reference
            end
            
            numCols = 1:nCols;
            numCols(refCol) = []; % Isolates the numerators
            dataTransformed = log(data(:, numCols) ./ data(:, refCol));
            
        case 'logit'
            % Logit: ln(p / (1 - p))
            % Bounds data to avoid infinities
            data = max(min(data, 1 - options.Offset), options.Offset);
            dataTransformed = log(data ./ (1 - data));
            
        case 'zscore'
            dataTransformed = (data - mean(data, 1, 'omitnan')) ./ std(data, 0, 1, 'omitnan');
            
        case 'log'
            dataTransformed = log(data + options.Offset);
            
    end
end