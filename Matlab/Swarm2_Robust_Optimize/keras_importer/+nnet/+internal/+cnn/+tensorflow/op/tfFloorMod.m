function z = tfFloorMod(x, y)

%   Copyright 2023 The MathWorks, Inc.

    xVal = x.value; 
    yVal = y.value;

    xRank = x.rank; 
    yRank = y.rank; 
    zRank = max(xRank, yRank); 
    
    if ~isfloat(xVal) && isfloat(yVal)
        xVal = cast(xVal, 'like', yVal); 
    elseif ~isfloat(yVal) && isfloat(xVal) 
        yVal = cast(yVal, 'like', xVal); 
    end 

    if isdlarray(xVal)
        xVal = extractdata(xVal);
    end

    if isdlarray(yVal)
        yVal = extractdata(yVal);
    end
    
    % in MATLAB, broadcasting starts from the left. In TF, broadcasting 
    % starts from the right. For this reason, we will keep x and y in 
    % the reverse TF dimension. 
    zVal = mod(xVal, yVal);

    if any(yVal==0, 'all')
        % mod(X,0) is X but FloorMod(X,0) is NaN
        % replace output by NaN wherever divisor is 0
        zRem = rem(xVal, yVal);
        zVal(isnan(zRem)) = NaN;
    end

    z = struct('value', zVal, 'rank', zRank); 
end 
