function result = translateTensorListReserve(~, ~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2023 The MathWorks, Inc.
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;

    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfTensorListReserve", ...
                    {MATLABOutputName}, MATLABArgIdentifierNames);
    result.NumOutputs = 1; 
    result.OpFunctions = "tfTensorListReserve";
    result.Success = true; 
end