function y = tfSplit(axis, value, numSplit)
    % Copyright 2022-2023 The MathWorks, Inc.

    valueVal = value.value; 
    valueRank = value.rank; 
    axisVal = axis.value;

    if isempty(axisVal)
    mlAxis = valueRank;
    else
        if axisVal < 0
           % Handle negative axis values
           axisVal = valueRank + 1 + axisVal;
        end
        mlAxis = valueRank - axisVal; 
    end 
    
    nSpecs = valueRank;
    idxArray = cell(1,numSplit);
    currSplitIdx = 1;
    splitSize = size(valueVal,mlAxis) / numSplit; 
    for i = 1:numSplit
        splitidxs = cell(1,nSpecs);
        for j = 1:nSpecs
            if (j==mlAxis)
                splitidxs{j} = currSplitIdx:(splitSize * i);
                currSplitIdx = currSplitIdx + splitSize;
            else
                splitidxs{j} = 1:size(valueVal,j);
            end
        end
        idxArray{i} = splitidxs;
    end

    y = cell(1,numel(idxArray));
    for k = 1:numel(idxArray)
        yVal = valueVal(idxArray{k}{:});       
        yVal = dlarray(single(yVal)); 
        y{k} = struct('value', yVal, 'rank', valueRank); 
    end     
end