function [y] = tfResize(image, outputSize, halfPixelCenters, method)

%   Copyright 2023 The MathWorks, Inc.

    imageRank = image.rank;
    imageVal = image.value;

    outputSizeVal = outputSize.value;

    % Input verification
    if imageRank ~= 4
        error('Resize is only supported for input tensors having a rank of 4.');
    end

    % Process outputSize
    outputSizeVal = single(outputSizeVal(:)');

    % Logic for the default "NHWC" data format
    imageVal = permute(imageVal, 4:-1:1);

    % Apply dlresize using forward TF format ordering
    if strcmp(method, 'linear')
        yVal = dlresize(imageVal, 'OutputSize', outputSizeVal, 'DataFormat', 'BSSC', 'GeometricTransformMode', halfPixelCenters, 'Method', 'linear');
    elseif strcmp(method, 'nearest_neighbor')
        yVal = dlresize(imageVal, 'OutputSize', outputSizeVal, 'DataFormat', 'BSSC', 'GeometricTransformMode', halfPixelCenters);
    end

    % Permute to reverse TF format ordering
    yVal = permute(yVal, 4:-1:1);

    % assign output rank:
    y = struct('value', yVal, 'rank', 4);

end
