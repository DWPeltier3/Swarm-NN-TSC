function result = translateTranspose(~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2022-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfTranspose", {MATLABOutputName}, MATLABArgIdentifierNames);
    
    result.Success = true;
    result.OpFunctions = "tfTranspose";
end 
