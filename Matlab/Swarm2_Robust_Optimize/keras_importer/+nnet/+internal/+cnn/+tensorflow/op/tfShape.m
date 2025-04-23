function y = tfShape(in)

%   Copyright 2020-2023 The MathWorks, Inc.
    
    inVal = in.value;    
    inRank = in.rank;

    shapeVal = size(inVal)'; 

    if inRank == 0
        % handle 0-D tensors.. 
        shapeVal = [];
    elseif inRank == 1
        % handle 1-D tensors. 
        shapeVal = shapeVal(1:inRank);
    elseif inRank > numel(shapeVal)
        % handle dropped trailing singleton dimensions. 
        rankDiff = inRank - numel(shapeVal); 
        shapeVal(end+1:end+(rankDiff)) = 1; 
    end 

    % tfShape gives output with Rank=1. Therefore the output should be in
    % forward TensorFlow format ordering
    if inRank > 1
        % Rank-1 input tensors are already in forward TF
        shapeVal = flip(shapeVal);
    end

    shapeVal = dlarray(shapeVal);
    y = struct('value', shapeVal, 'rank', 1);
end
