function y = tfRange(start, limit, delta)

%   Copyright 2020-2023 The MathWorks, Inc.

    startVal = start.value; 
    limitVal = limit.value; 
    deltaVal = delta.value;

    if deltaVal >= 0
        yVal = single(startVal):single(deltaVal):single(limitVal-1); 
        yVal = cast(yVal, 'like', startVal); 
    else 
        yVal = single(startVal):single(deltaVal):single(limitVal+1); 
        yVal = cast(yVal, 'like', startVal); 
    end
    if ~isfloat(yVal)
        yVal = cast(yVal, 'single'); 
    end 
    y = struct('value', dlarray(yVal'), 'rank', 1);
end 
