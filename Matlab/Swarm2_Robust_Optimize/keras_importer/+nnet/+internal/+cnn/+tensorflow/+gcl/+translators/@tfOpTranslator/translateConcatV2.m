function result = translateConcatV2(this, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfCat", MATLABOutputName, [[MATLABArgIdentifierNames(end)] MATLABArgIdentifierNames(1:end - 1)]);
    result.OpFunctions = "tfCat";
    result.NumOutputs = 1;
    result.Success = true;
end
