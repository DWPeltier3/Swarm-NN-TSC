function result = translateProd(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2021-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    if isfield(node_def.attr, 'keep_dims')
        keepdims = node_def.attr.keep_dims.b;
    else
        keepdims = false;
    end
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfProd", MATLABOutputName, ...
        [MATLABArgIdentifierNames {keepdims}]); 
    
    result.OpFunctions = "tfProd";
    result.NumOutputs = 1; 
    result.Success = true; 
end 
