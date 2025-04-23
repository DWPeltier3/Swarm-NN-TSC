function y = tfSlice(input, begin, sliceSize)

    % Copyright 2022-2024 The MathWorks, Inc.

    inputVal = input.value; 
    inputRank = input.rank; 
    beginVal = begin.value; 
    beginRank = begin.rank;
    sliceSizeVal = sliceSize.value; 
    sliceSizeRank = sliceSize.rank; 

    % Get the values and ensure the ranks of begin and size are the same
    assert(beginRank == sliceSizeRank, "tfSlice: ranks of begin and size are different.");
    assert(numel(beginVal) == numel(sliceSizeVal), "tfSlice: shape of begin and size are different.");
    nSpecs = numel(beginVal);

    % Input is in reverse TF format
    % Flip the index vectors to match the input
    beginVal = flip(beginVal);
    sliceSizeVal = flip(sliceSizeVal);

    S = substruct('()', {});
    for i = 1:nSpecs
        curBeginIdx = single(beginVal(i) + 1);
        if sliceSizeVal(i) == -1
            curEndIdx = single(size(inputVal, i));
        else
            curEndIdx = single(beginVal(i)) + single(sliceSizeVal(i));
        end
        idxVec = curBeginIdx:curEndIdx; 
        S.subs{i} = idxVec;
    end 
    
    % Index into the Input array, preserve the number of dimensions.
    yVal = inputVal(S.subs{:});
    y = struct('value', yVal, 'rank', inputRank); 
end
