function result = translateLogicalAnd(~, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    result.Code = MATLABOutputName + ".value" + " = iExtractData(" + MATLABArgIdentifierNames{1} + ".value)" + " & iExtractData(" + MATLABArgIdentifierNames{2} + ".value);";  
    result.Code = result.Code + newline + MATLABOutputName + ".value = " + "cast(" + MATLABOutputName + ".value, 'like', " + MATLABArgIdentifierNames{1} + ".value);";
    result.ForwardRank = true; 
    result.Success = true; 
end 