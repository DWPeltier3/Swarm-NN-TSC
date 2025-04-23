function result = translateWhere(~, ~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2023 The MathWorks, Inc.
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfWhere", ...
                    {MATLABOutputName}, MATLABArgIdentifierNames);

    result.NumOutputs = 1; 
    result.OpFunctions = "tfWhere";
    result.Success = true; 
end