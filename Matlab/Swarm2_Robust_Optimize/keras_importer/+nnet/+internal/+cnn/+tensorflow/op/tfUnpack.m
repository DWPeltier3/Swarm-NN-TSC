function varargout = tfUnpack(value, num, axis)

% Copyright 2022-2023 The MathWorks, Inc.

% num and TFAxis are scalar node attributes, they should never be structs
    
    xVal = value.value;
    xRank = value.rank;

    assert(xRank >= 1, "tfUnpack: value rank is less than 1.");

    % xVal should be in reverse TF format.    
    if axis < 0
        % handle negative axis values
        mlAxis = mod(axis, xRank);
    else 
        mlAxis = axis; 
    end
    mlAxis = xRank - mlAxis;
    
    numOuts = floor(size(xVal, mlAxis)  / num); 
    start = 1; 
    varargout = cell(1, numOuts); 
    outShape = ones(1, xRank); 
    outShape(1:ndims(xVal)) = size(xVal);
    outShape(mlAxis) = []; 

    for i = 1:num
        indices = repmat({':'}, [1 xRank]); 
        indices{mlAxis} = i;
        start = start + num; 
        varargout{i}.value = xVal(indices{:}); 
        if numel(outShape) > 1
            varargout{i}.value = reshape(varargout{i}.value, outShape);  
        end
        varargout{i}.rank = xRank - 1; 
        varargout{i}.value = dlarray(varargout{i}.value);
    end
end