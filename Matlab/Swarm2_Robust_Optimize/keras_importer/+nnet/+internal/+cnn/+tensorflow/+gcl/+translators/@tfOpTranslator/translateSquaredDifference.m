function result = translateSquaredDifference(~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    call = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfSub", {MATLABOutputName}, MATLABArgIdentifierNames);
    call = call + newline + nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfMul", {MATLABOutputName}, {MATLABOutputName, MATLABOutputName}); 
    result.Code = call; 
    
    result.OpFunctions = ["tfMul" "tfSub"]; 
    result.Success = true; 
end 
