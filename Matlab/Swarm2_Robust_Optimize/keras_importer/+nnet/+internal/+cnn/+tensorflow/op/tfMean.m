function y = tfMean(x, axis, keepDims)

%   Copyright 2020-2024 The MathWorks, Inc.

    xVal = x.value;
    xRank = x.rank;

    % If axis is a struct extract the numeric value
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
    
    % Apply mean, this preserves all dimensions
    yVal = mean(xVal, MLAxis); 
    
    if ~keepDims
        outsize = ones(1, xRank); 
        outsize(1:ndims(yVal)) = size(yVal); 
        outsize(MLAxis) = [];
        if numel(outsize) < 1
            outsize = [1 1];
        end
        yRank = xRank - numel(MLAxis); 
        
        % Reshape to the reduced dims and set all labels to U
        if yRank > 1   
            yVal = reshape(yVal, outsize);
        else
            yVal = reshape(yVal, [outsize 1]); 
        end       
    end

    y = struct('value', yVal, 'rank', yRank);
end
