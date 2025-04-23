function result = translatePadV2(~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfPad", {MATLABOutputName}, MATLABArgIdentifierNames);
    result.OpFunctions = "tfPad"; 
    result.Success = true; 
end 
