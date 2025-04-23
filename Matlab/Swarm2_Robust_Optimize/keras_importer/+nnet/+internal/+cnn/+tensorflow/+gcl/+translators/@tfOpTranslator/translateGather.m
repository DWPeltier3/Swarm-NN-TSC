function result = translateGather(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
    %   Copyright 2023 The MathWorks, Inc.
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    batch_dims = 0;
    if isfield(node_def.attr, 'batch_dims')
        batch_dims = str2double(node_def.attr.batch_dims.i); 
    end
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfGather", ...
        {MATLABOutputName}, [MATLABArgIdentifierNames {batch_dims}]);
    
    result.NumOutputs = 1;
    result.OpFunctions = "tfGather";
    result.Success = true; 
end