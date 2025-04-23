function result = translateStack(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    
    axis = "[]"; 
    if isfield(node_def.attr, 'axis') 
        axis = node_def.attr.axis.i;
    end 
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfStack", {MATLABOutputName}, [{axis}, MATLABArgIdentifierNames]); 
    result.NumOutputs = 1; 
    result.OpFunctions = "tfStack"; 
    result.Success = true; 
end 
