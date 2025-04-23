function result = translateInlinedUnaryOpLambdaOp(~, fcnStr, MATLABOutputName, MATLABArgIdentifierNames)
% 

%   Copyright 2023 The MathWorks, Inc.
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    result.Code = MATLABOutputName + " = struct('value', " + fcnStr + "(" + MATLABArgIdentifierNames{1} + ".value), 'rank', "+ MATLABArgIdentifierNames{1} + ".rank);";
    result.NumOutputs = 1;
    result.Success = true;   
end 
