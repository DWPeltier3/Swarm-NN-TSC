function result = translateNeg(~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2021 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    result.Code = MATLABOutputName + ".value" + " = -" + MATLABArgIdentifierNames{1} + ".value" + ";"; 
    result.ForwardRank = true;
    result.Success = true; 
end 
