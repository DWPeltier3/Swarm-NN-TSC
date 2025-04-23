function y = tfBiasAdd(value, bias, dataFormat)

    %   Copyright 2020-2023 The MathWorks, Inc.
    
    valueVal = value.value;
    valueRank = value.rank;
    biasVal = bias.value;
        
    % If the input tensor is unlabeled. We assume it is in reverse TensorFlow order
    if strcmp(dataFormat, "NHWC")
        % In the reverse TF format, first dimension is channels
        channelDim = 1;
    elseif strcmp(dataFormat, "NCHW")
        % In the reverse TF format, second last dimension is channels
        channelDim = valueRank - 1;
    else
        error('BiasAdd only supports input data formats: "NHWC" and "NCHW"');
    end
    
    biasShape = num2cell(ones(valueRank, 1)); 
    biasShape{channelDim} = []; 

    y =valueVal + reshape(biasVal, biasShape{:}); 
    y = struct('value', y, 'rank', valueRank);
end
