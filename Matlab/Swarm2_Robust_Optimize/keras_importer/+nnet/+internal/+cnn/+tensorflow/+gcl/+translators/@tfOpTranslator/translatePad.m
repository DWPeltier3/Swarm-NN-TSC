function result = translatePad(~, MATLABOutputName, MATLABArgIdentifierNames)
    % Pad is the same as PadV2 but with only zero padding. 

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfPad", {MATLABOutputName}, [MATLABArgIdentifierNames {'0'}]);
    result.OpFunctions = "tfPad"; 
    result.Success = true; 
end 
