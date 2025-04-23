function [y] = tfMaxPool(input, kSize, strides, padding, explicitPaddings)

%   Copyright 2020-2023 The MathWorks, Inc.

    inputRank = input.rank;
    inputVal = input.value;

    % Input verification
    if inputRank ~= 4
        error('Max pooling is only supported for input tensors having a rank of 4.');
    end

    % Logic for the default "NHWC" data format

    inputVal = permute(inputVal, 4:-1:1);
    
    % In TF ksize and strides are 1-D Tensors of size 4 (NHWC).
    % Extract the dimensions corresponding to H and W (index 2 and 3)
    % Flip to convert to reverse TF order
    kSize = kSize([2 3]);
    strides = strides([2 3]);

    % In TF explicit padding is a list of size 8 (2 values per dimension)
    % Extract the padding values corresponding to the spatial dimensions (index 2 and 3)
    % Flip to convert to reverse TF order
    if ~isempty(explicitPaddings)    
        explicitPaddings = explicitPaddings(3:6);
        % convert padding from TF: [top bottom left right] 
        % to MATLAB: [top, left; bottom, right]
        explicitPaddings = reshape(explicitPaddings,[2,2]);
    end
    
    if strcmp(padding,'VALID')
        yVal = maxpool(inputVal, kSize, 'DataFormat', 'BSSC', 'Stride', strides);
    elseif strcmp(padding,'SAME')
        yVal = maxpool(inputVal,  kSize, 'DataFormat', 'BSSC', 'Stride', strides,'Padding','same');
    elseif strcmp(padding,'EXPLICIT')
        yVal = maxpool(inputVal, kSize, 'DataFormat', 'BSSC', 'Stride', strides, 'Padding', explicitPaddings);
    end

    yVal = permute(yVal, 4:-1:1);

    % assign output rank:
    y = struct('value', yVal, 'rank', 4);

end

