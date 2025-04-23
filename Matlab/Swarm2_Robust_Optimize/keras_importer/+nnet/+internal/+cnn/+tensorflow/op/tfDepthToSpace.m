function [y] = tfDepthToSpace(input, blockSize)

%   Copyright 2023 The MathWorks, Inc.

    inputRank = input.rank;
    inputVal = input.value;

    % Input verification
    if inputRank ~= 4
        error('DepthToSpace is only supported for input tensors having a rank of 4.');
    end

    % Logic for the default "NHWC" data format
    inputVal = permute(inputVal, 4:-1:1);

    % Apply depthToSpace using forward TF format ordering
    yVal = depthToSpace(inputVal, blockSize, 'DataFormat', 'BSSC');

    % Permute to reverse TF format ordering
    yVal = permute(yVal, 4:-1:1);

    % assign output rank:
    y = struct('value', yVal, 'rank', 4);

end