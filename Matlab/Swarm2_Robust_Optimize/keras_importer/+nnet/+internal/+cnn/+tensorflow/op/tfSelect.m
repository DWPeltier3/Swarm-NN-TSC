function z = tfSelect(condition, x, y)

%   Copyright 2022-2023 The MathWorks, Inc.

    xRank = x.rank; 
    xVal = x.value; 
    yRank = y.rank;
    yVal = y.value; 
    cRank = condition.rank;
    cVal = condition.value;
    
    % x and y should have the same rank (and shape)
    assert(xRank == yRank, "tfSelect: ranks of x and y are different.");
    assert(cRank == xRank, "tfSelect: ranks of condition and x are different. This is not currently supported.");
    zRank = xRank;
        
    if xRank == 0 % assume y and condition are scalars
        if cVal
            z.value = xVal;
        else
            z.value = yVal;
        end
        z.rank = zRank; 
        return; 
    end 

    % xval, yval and cval should be in reverse Tf format at this point.
    if isequal(numel(size(cVal)),numel(size(xVal)))
        % condition has the same shape as x and y
        zVal = xVal;
        zVal(~cVal) = yVal(~cVal);
    else        
        % If condition is rank 1, x may have higher rank, 
        % but its first dimension must match the size of condition. This is not currently supported.       
        warning('tfSelect: ranks of condition and x are different. This is not currently supported.');
    end

    z = struct('value', zVal, 'rank', zRank); 
end 
