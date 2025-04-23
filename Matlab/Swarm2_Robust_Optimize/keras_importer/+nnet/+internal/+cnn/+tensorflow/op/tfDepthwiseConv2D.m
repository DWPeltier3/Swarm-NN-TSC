function [y] = tfDepthwiseConv2D(input, filter, strides, padding, explicitPaddings, dilations)

%   Copyright 2020-2023 The MathWorks, Inc.

    inputRank = input.rank;
    inputVal = input.value;
    filterVal = filter.value;

    % Input verification
    if inputRank ~= 4
        error('2D depthwise convolution is only supported for input tensors having a rank of 4.');
    end
    
    % Logic for the default "NHWC" data format
    % The TF DepthwiseConv2dNative op currently only supports the NHWC tensor format on the CPU.

    % In TF strides and dilations are 1-D Tensors of size 4 (NHWC).
    % Extract the dimensions corresponding to H and W (index 2 and 3)
    strides = fliplr(strides([2 3]));
    
    if ~isempty(dilations)
        dilations = fliplr(dilations([2 3]));
    else
        dilations = 1;
    end
    
    % TF DepthwiseConv2D raw op doesn't have a bias
    bias = 0;
    
    % In TF explicit padding is a list of size 8 (2 values per dimension)
    % Extract the padding values corresponding to the spatial dimensions (index 2 and 3)
    if ~isempty(explicitPaddings)    
        explicitPaddings = explicitPaddings(3:6);
        % convert [top bottom left right] to [top, left; bottom, right]
        explicitPaddings = fliplr(reshape(explicitPaddings,[2,2]));
    end

    % shape and label filter
    [channelMultiplier, inChannels, width, height] = size(filterVal);
    filterVal = permute(filterVal, [3 4 1 2]); 
    filterVal = reshape(filterVal, [width, height, 1, channelMultiplier, inChannels]);
    % dlconv > size(weights) = filtersize (SS dimension), numChannelsPerGroup (C dimension),
    % numFiltersPerGroup (first U dimension), numGroups (second U dimension)
    % for depthwise 2D convolution:
    % numChannelsPerGroup = 1, numFiltersPerGroup = channel_multiplier,
    % numGroups = in_channels
    filterVal= dlarray(filterVal,'SSCUU');
    
    % Applying dlconv on formatted dlarray
    if strcmp(padding,'VALID')
        yVal = dlconv(inputVal, filterVal, bias, 'DataFormat', 'CSSB', 'Stride', strides,'DilationFactor', dilations);
    elseif strcmp(padding,'SAME')
        yVal = dlconv(inputVal, filterVal, bias, 'DataFormat', 'CSSB', 'Stride', strides, 'Padding', 'same', 'DilationFactor', dilations);
    elseif strcmp(padding,'EXPLICIT')
        yVal = dlconv(inputVal, filterVal, bias, 'DataFormat', 'CSSB', 'Stride', strides, 'Padding', explicitPaddings, 'DilationFactor', dilations);
    end

    % assign output rank
    y = struct('value', yVal, 'rank', 4);

end

