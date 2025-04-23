function [y] = tfAvgPool(value, kSize, strides, padding)

%   Copyright 2020-2023 The MathWorks, Inc.
    valueVal = value.value;
    valueRank = value.rank;

    % Input verification
    if valueRank ~= 4
        error('Average pooling is only supported for input tensors having a rank of 4.');
    end
    
    % Permute to forward TF format
    valueVal = permute(valueVal, 4:-1:1);

    % Logic for the default "NHWC" data format
    % In TF ksize and strides are 1-D Tensors of size 4 (NHWC).
    % Extract the dimensions corresponding to H and W (index 2 and 3)
    % Flip to convert to reverse TF order
    kSize = kSize([2 3]);
    strides = strides([2 3]);

    if strcmp(padding,'VALID')
        yVal = avgpool(valueVal, kSize, 'DataFormat', 'BSSC', 'Stride', strides);
    elseif strcmp(padding,'SAME')
        yVal = avgpool(valueVal, kSize, 'DataFormat', 'BSSC', 'Stride', strides, 'Padding', 'same', 'PaddingValue', 'mean');
    end

    % Permute to Reverse TF format
    yVal = permute(yVal, 4:-1:1);
    
    % assign output rank:
    y = struct('value', yVal, 'rank', 4);
end

