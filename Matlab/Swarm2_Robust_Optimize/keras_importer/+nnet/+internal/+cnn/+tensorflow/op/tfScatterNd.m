function y = tfScatterNd(indices, updates, shape)
   
%   Copyright 2023-2024 The MathWorks, Inc.

    indices = convertToForwardTF(indices);
    updates = convertToForwardTF(updates);
    shape = convertToForwardTF(shape);

    idxVal = indices.value;
    updatesVal = updates.value;
    updatesRank = updates.rank;
    shapeVal = shape.value;
    
    updatesShape = size(updatesVal);
    if (numel(updatesShape)) < updatesRank
        % Add back dropped singletons
        numDroppedDims = updatesRank - numel(updatesShape);
        updatesShape(end+1:end+numDroppedDims) = 1;
    elseif (numel(updatesShape)) > updatesRank
        % only in case of rank 1
        updatesShape(end) = [];
    end
    
    yRank = numel(shapeVal);

    if numel(updatesShape) == numel(shapeVal') && all(updatesShape == shapeVal')
        % Output same as updates
        yVal = updatesVal;
    else    
        if yRank <=1
            yVal = zeros([shapeVal 1]);
        else
            yVal = zeros(shapeVal');
        end    
        nSpecs = size(idxVal,1);
        idxVal = idxVal + 1;    
        for i = 1:nSpecs
            curIdxVal = num2cell(idxVal(i,:));
            if numel(curIdxVal) == yRank
                % Fully specified indices
                yVal(curIdxVal{:}) = updatesVal(i);
            else
                % Partially specified indices
                numIdx = numel(curIdxVal);
                numShape = numel(shapeVal);
                extraDimsNeeded = numShape - numIdx;
                curIdxVal(end+1:end+extraDimsNeeded) = {':'};            
                curUpdatesIdxVal = {i};
                curUpdatesIdxVal(end+1:end+extraDimsNeeded) = {':'};
                yVal(curIdxVal{:}) = updatesVal(curUpdatesIdxVal{:});
            end
        end
    end
    
    if yRank > 1
        % permute to reverse TF
        yVal = permute(yVal, yRank:-1:1);
    end
    yVal = dlarray(yVal);
    y = struct('value', yVal, 'rank', yRank); 
end

function [x] = convertToForwardTF(x)
    if x.rank > 1
        x.value = permute(x.value, x.rank:-1:1);
    end
end 