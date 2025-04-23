function result = translateCast(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfCast", MATLABOutputName, [MATLABArgIdentifierNames {"'" + node_def.attr.DstT.type + "'"}]);
    result.Success = true;
    result.OpFunctions = "tfCast";
end 
