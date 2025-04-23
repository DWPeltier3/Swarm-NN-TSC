function y = tfRandomStandardNormal(shape, dtype)

%   Copyright 2020-2023 The MathWorks, Inc.

    shapeVal = shape.value;
    yRank = numel(shapeVal); 
    
    if isa(shapeVal, 'dlarray')
        shapeVal = shapeVal.extractdata; 
    end 
    % Force it to be row vec. 
    shapeVal = fliplr(shapeVal(:)'); 
    if numel(shapeVal) == 1
        shapeVal = [shapeVal 1]; 
    end 
    switch dtype
        case 'DT_FLOAT'
            yVal = randn(shapeVal, 'single'); 
        case 'DT_DOUBLE'
            yVal = randn(shapeVal, 'double'); 
        otherwise 
            yVal = randn(shapeVal, 'single'); 
    end
    
    y = struct('value', yVal, 'rank', yRank);
end
