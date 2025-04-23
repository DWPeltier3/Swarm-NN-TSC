function y = tfExpandDims(x, axis)

%   Copyright 2021-2024 The MathWorks, Inc.

    xVal = x.value; 
    xRank = x.rank; 
    
    % If axis is a struct extract the numeric axis value
    if isstruct(axis)
        axisVal = axis.value;
    else
        % if input is numeric
        axisVal = axis;
    end

	% Handle negative axis value    
    if axisVal < 0
		axisVal = xRank + axisVal + 1;
    end
    
    yRank = xRank + 1;
    
    % At this point the input should be in reversed TF format
    % The expanded dimension should be added in the forward TF dim order
    % Hence, the reshape vector needs to be in forward TF format first
    % Then we add the expanded singleton dimension to it
    % Then we flip it again to convert to reverse TF format, before using it for reshape
    if xRank > 1
        reshapeVector = ones(1, xRank);    % [1 1 1 1] , size(X.value) : [2 4 3 1] (in reverse tf format)
        reshapeVector(:, 1:ndims(xVal)) = size(xVal); % [2 4 3 1]
        reshapeVector = flip(reshapeVector); % [1 3 4 2] (forward tf format)
        reshapeVector = [reshapeVector(1:axisVal) 1 reshapeVector(axisVal+1:end)];  % [1 3 1 4 2]
        reshapeVector = flip(reshapeVector); % [2 4 1 3 1] (reverse tf format)
        yVal = reshape(xVal, reshapeVector);
    else
        % rank=1 in TF is already represented as rank=2 in MATLAB
        yVal = xVal;
        if axisVal ~= 0
            yVal = permute(yVal, yRank:-1:1);
        end
    end    
    
    yVal = dlarray(yVal);
    y = struct('value', yVal, 'rank', yRank);
end