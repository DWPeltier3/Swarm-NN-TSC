function y = tfArgMin(input, dimension)

%   Copyright 2023-2024 The MathWorks, Inc.

    xVal = input.value; 
    xRank = input.rank;

    if isstruct(dimension)
        dimension = dimension.value;
    end

    % Handle negative dimension or "axis" values
    if dimension < 0
       dimension = xRank + dimension;
    end
    
    % Reverse TF dimension order
    MLAxis = (xRank - dimension);
    if input.rank <= 1
        [~,y] = min(xVal(:), [], MLAxis); 
    else 
        [~,y] = min(xVal, [], MLAxis); 
    end 
    
    % Change to zero-based indexing
    y = y - 1;
    
    dimsToDrop = MLAxis;
    dimsToDrop(dimsToDrop > ndims(y)) = [];
    
    newSize = size(y);
    newSize(dimsToDrop)= [];
    if numel(newSize) == 1
        y = reshape(y,newSize, []);
    elseif numel(newSize) > 1
        y = reshape(y,newSize);
    end

    yRank = xRank-numel(dimension);
    y = struct('value', y, 'rank', yRank);
end 
