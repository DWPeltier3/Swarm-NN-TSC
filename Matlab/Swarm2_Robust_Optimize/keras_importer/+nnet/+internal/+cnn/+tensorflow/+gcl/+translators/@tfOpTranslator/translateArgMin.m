function result = translateArgMin(~, ~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfArgMin", MATLABOutputName, MATLABArgIdentifierNames); 
    
    result.OpFunctions = "tfArgMin";
    result.NumOutputs = 1; 
    result.Success = true; 
end 
