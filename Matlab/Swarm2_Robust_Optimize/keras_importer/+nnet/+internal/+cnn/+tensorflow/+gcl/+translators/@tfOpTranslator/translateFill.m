function result = translateFill(~, MATLABOutputName, MATLABArgIdentifierNames)
    % Copyright 2021-2023 The MathWorks, Inc.
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 

    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfFill", ...
        MATLABOutputName, MATLABArgIdentifierNames);

    result.OpFunctions = "tfFill";
    result.NumOutputs = 1;
    result.Success = true; 
end 
