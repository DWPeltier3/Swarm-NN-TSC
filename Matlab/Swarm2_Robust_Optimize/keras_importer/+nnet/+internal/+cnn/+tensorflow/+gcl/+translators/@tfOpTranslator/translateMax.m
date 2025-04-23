function result = translateMax(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    if isfield(node_def.attr, 'keep_dims')
        keepdims = node_def.attr.keep_dims.b;
    else
        keepdims = false;
    end
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfMax", MATLABOutputName, ...
        [MATLABArgIdentifierNames {keepdims}]); 
    
    result.OpFunctions = "tfMax";
    result.NumOutputs = 1; 
    result.Success = true; 
end 
