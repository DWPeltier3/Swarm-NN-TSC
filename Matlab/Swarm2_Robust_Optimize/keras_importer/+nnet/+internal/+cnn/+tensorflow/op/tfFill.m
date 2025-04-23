function y = tfFill(dims, value)

    % Copyright 2022-2024 The MathWorks, Inc.

    dimsVal = dims.value;
    dimsRank = dims.rank;

    valueVal = value.value;
    valueRank = value.rank;

    % Get the values and ensure they are in a row vector. 
    assert(dimsRank <= 1, "tfFill: dims is not 1D.");
    assert(valueRank == 0, "tfFill: value to fill is not a scalar.")

    % convert dims to a row vector
    dimsVal = dimsVal(:)'; 
    
    yRank = numel(dimsVal);
    
    if yRank == 1
    % Shape of 1D tensors is kept as is
    % but here we reverse this shape
    % so that the logic below reverses it again
    % to be same as that in TF
        dimsVal = [1 dimsVal];
    end
    
    % generate the ones to have the same shape as reverse tensorflow
    yVal = ones(dimsVal(end:-1:1), 'like', valueVal) .* valueVal;
    yVal = dlarray(yVal);
    
    y = struct('value', yVal, 'rank', yRank);
end
