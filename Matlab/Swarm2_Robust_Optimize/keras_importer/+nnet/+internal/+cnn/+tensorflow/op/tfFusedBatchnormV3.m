function [y, batchMean, batchVariance] = tfFusedBatchnormV3(...
    x, scale, offset, mean, variance, isTraining, epsilon, dataFormat)
    %{{import_statement}}
    
%   Copyright 2020-2023 The MathWorks, Inc.
    % FusedBatchNormV3
    xRank = x.rank;
    xVal = x.value;
    offsetVal = offset.value; 
    scaleVal = scale.value; 
    meanVal = mean.value; 
    varianceVal = variance.value; 

    % Input verification
    if ~ismember(xRank, [4 5])
        error('FusedBatchNormV3 is only supported for input tensors having a rank of 4 or 5.');
    end
    
    if ~ismember(dataFormat, ["NHWC", "NCHW", "NDHWC"])
        error('FusedBatchNormV3 is only supported for input data formats: "NHWC", "NCHW" and "NDHWC"');
    end

    switch dataFormat
            case "NHWC"
                tfxLabels = "BSSC";
            case "NCHW"
                tfxLabels = "BCSS";
            case "NDHWC"
                tfxLabels = "BSSSC";            
    end
       
    % Permute to forward tensorflow format and apply labels
    xVal = permute(stripdims(xVal), xRank:-1:1); 
    xVal = dlarray(xVal, tfxLabels);     
    if isTraining
        [y, batchMean, batchVariance] = batchnorm(xVal, offsetVal, scaleVal, meanVal, varianceVal, 'Epsilon', epsilon); 
    else
        y = batchnorm(xVal, offsetVal, scaleVal, meanVal, varianceVal, 'Epsilon', epsilon); 
        batchMean = meanVal;
        batchVariance = varianceVal;
    end
    
    % Permutes to reverse tensorflow
    y = iPermuteToReverseTF(y, xRank);

    % forward the rank. 
    y = struct('value', y, 'rank', xRank); 
    if nargout > 1 
        batchMean = struct('value', batchMean, 'rank', 1); 
        batchVariance = struct('value', batchVariance, 'rank', 1); 
    end
end
