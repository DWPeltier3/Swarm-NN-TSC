function result = translateFusedBatchNorm(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames, multiOutputNameMap)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 

    if numOutputs > 1
        % this op has 6 TF outputs. We are only able to use
        % the first 3: output, updated mean, and updated
        % variance. The other two outputs shouldn't be used
        % in inference. 
        outputNames = makeMultipleOutputArgs(this, MATLABOutputName, 3); 
    else
        outputNames = {MATLABOutputName}; 
    end
    
    if isfield(node_def.attr, 'data_format') && isfield(node_def.attr.data_format, 's')
        dataFormat = char(matlab.net.base64decode(node_def.attr.data_format.s));
    else
        dataFormat = 'NHWC';
    end
    
    if isfield(node_def.attr, 'epsilon') && isfield(node_def.attr.epsilon, 'f')
        epsilon = {num2str(max(node_def.attr.epsilon.f, 1e-5))};  % batchnorm only supports epsilon values >= 1e-5
    else 
        epsilon = {num2str(0.0001)};
    end

    isTraining = [char(this.LAYERREF) '.IsTraining'];    
    
    meanName = MATLABArgIdentifierNames(4);
    varName  = MATLABArgIdentifierNames(5);
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfFusedBatchnormV3", outputNames, [MATLABArgIdentifierNames([1 2 3]) meanName varName isTraining epsilon ['"' dataFormat '"']]);
    multiOutputNameMap(node_def.name) = {'y', 'batch_mean', 'batch_variance', 'reserve_space_1', 'reserve_space_2', 'reserve_space_3'}; %#ok<NASGU>

    result.Success = true;
    result.OpFunctions = "tfFusedBatchnormV3";
end 
