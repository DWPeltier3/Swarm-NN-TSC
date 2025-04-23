function z = tfFloorDiv(x, y)

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
    
    % in MATLAB, broadcasting starts from the left. In TF, broadcasting 
    % starts from the right. For this reason, we will keep x and y in 
    % the reverse TF dimension. 
    zVal = floor(rdivide(xVal, yVal));

    z = struct('value', zVal, 'rank', zRank); 
end 
