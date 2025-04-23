function result = translateSelect(~, ~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2022-2023 The MathWorks, Inc.
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfSelect", ...
                    {MATLABOutputName}, MATLABArgIdentifierNames);
    result.NumOutputs = 1; 
    result.OpFunctions = "tfSelect";
    result.Success = true; 
end