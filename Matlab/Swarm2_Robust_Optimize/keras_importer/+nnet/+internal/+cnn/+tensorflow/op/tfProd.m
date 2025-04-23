function y = tfProd(x, axis, keepDims)
    
    % Copyright 2022-2024 The MathWorks, Inc.

    xVal = x.value;
    xRank = x.rank;
    
    % If axis is a struct extract the numeric axis value
    if isstruct(axis)
        axisVal = axis.value;
    else
        % if input is numeric
        axisVal = axis;
    end
    
    yRank = xRank;

    if any(axisVal < 0)
        % Handle negative axis values
        negIdx = axisVal < 0;
        axisVal(negIdx) = xRank + axisVal(negIdx);
    end
    
    % xval is in reverse TF format
    MLAxis = xRank - axisVal;
    
    % Reverse TensorFlow dimension order
    if xRank <= 1
        yVal = prod(xVal(:), MLAxis); 
    else 
        yVal = prod(xVal, MLAxis); 
    end    

    if nargin < 3
        keepDims = false;
    end

    if ~keepDims
        dimsToDrop = MLAxis;
        dimsToDrop(dimsToDrop > ndims(yVal)) = [];    
        newSize = size(yVal);
        newSize(dimsToDrop)= [];

        yRank = xRank - numel(MLAxis);

        if numel(newSize) == 1
            yVal = reshape(yVal, newSize, []);
        elseif numel(newSize) > 1
            yVal = reshape(yVal, newSize);
        end
    end

    y = struct('value', yVal, 'rank', yRank);
end 
