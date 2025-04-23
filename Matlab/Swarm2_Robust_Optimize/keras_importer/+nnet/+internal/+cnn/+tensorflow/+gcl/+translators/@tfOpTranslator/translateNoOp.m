function result = translateNoOp(~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2022 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    result.Code = ""; 
    result.ForwardRank = false;
    result.Success = true;
end 
