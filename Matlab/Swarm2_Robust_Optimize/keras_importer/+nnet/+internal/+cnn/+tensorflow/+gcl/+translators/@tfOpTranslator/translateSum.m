function result = translateSum(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2021-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    
    keep_dims = "false"; 
    if isfield(node_def.attr, 'keep_dims')
        if node_def.attr.keep_dims.b
            keep_dims = "true"; 
        end
    end

    % output = tfSum(tensor, axis, keepdims); 
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfSum", ...
        MATLABOutputName, [MATLABArgIdentifierNames keep_dims]); 
    
    result.OpFunctions = "tfSum";
    result.NumOutputs = 1;
    result.Success = true;
end
