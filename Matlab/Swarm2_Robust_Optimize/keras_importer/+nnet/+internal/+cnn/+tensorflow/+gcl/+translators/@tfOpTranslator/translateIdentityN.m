function result = translateIdentityN(this, numOutputs, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    if numOutputs > 1
        outputNames = makeMultipleOutputArgs(this, MATLABOutputName, numOutputs); 
    else
        outputNames = {MATLABOutputName}; 
    end

    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfIdentityN", outputNames, MATLABArgIdentifierNames);
    result.OpFunctions = "tfIdentityN"; 
    result.Success = true;
end 
