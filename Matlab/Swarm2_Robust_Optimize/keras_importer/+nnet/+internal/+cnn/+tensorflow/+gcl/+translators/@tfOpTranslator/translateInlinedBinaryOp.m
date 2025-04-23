function result = translateInlinedBinaryOp(~, outputDims, fcnStr, isLogicalOp, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    if isLogicalOp
        result.Code = MATLABOutputName + " = " + "struct('value', cast(" + fcnStr + "(" + MATLABArgIdentifierNames{1} + ".value, " + ...
            MATLABArgIdentifierNames{2} + ".value), 'like', " + MATLABArgIdentifierNames{1} + ".value), 'rank', " + string(outputDims) + ");";
    else
        result.Code = MATLABOutputName + " = " + "struct('value', " + fcnStr + "(" + MATLABArgIdentifierNames{1} +  ...
            ".value, " + MATLABArgIdentifierNames{2} + ".value), 'rank', " + string(outputDims) + ");";
    end
        result.Success = true; 
end
