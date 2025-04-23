function result = translateExpandDims(~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2021-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfExpandDims", {MATLABOutputName}, MATLABArgIdentifierNames); 
    result.OpFunctions = "tfExpandDims"; 
    result.Success = true; 
end 
