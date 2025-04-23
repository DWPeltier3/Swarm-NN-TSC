function z = tfMinimum(x, y)

%   Copyright 2020-2023 The MathWorks, Inc.

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

    zVal = min(xVal, yVal); 

    z = struct('value', zVal, 'rank', zRank); 
end 
