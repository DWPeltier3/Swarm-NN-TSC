function result = translateAll(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 

    keep_dims = "false";
    if isfield(node_def.attr, 'keep_dims')
        if node_def.attr.keep_dims.b
            keep_dims = "true";
        end
    end
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfAll", {MATLABOutputName}, [MATLABArgIdentifierNames, keep_dims]); 
    result.NumOutputs = 1; 
    result.OpFunctions = "tfAll"; 
    result.Success = true; 
end 
