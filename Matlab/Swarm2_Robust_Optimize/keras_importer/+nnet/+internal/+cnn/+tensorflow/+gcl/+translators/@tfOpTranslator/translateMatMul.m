function result = translateMatMul(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    transp_a = false;
    transp_b = false;
    
    if isfield(node_def.attr,'transpose_a')
        transp_a = node_def.attr.transpose_a.b;
    end
    
    if isfield(node_def.attr,'transpose_b')
        transp_b = node_def.attr.transpose_b.b;
    end
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfMatMul", {MATLABOutputName}, ...
                    [MATLABArgIdentifierNames {transp_a, transp_b}]); 
    
    result.OpFunctions = "tfMatMul";
    result.Success = true; 
end 
