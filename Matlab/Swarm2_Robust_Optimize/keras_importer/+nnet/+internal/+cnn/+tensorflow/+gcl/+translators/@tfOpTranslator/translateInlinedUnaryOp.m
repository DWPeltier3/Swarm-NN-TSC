function result = translateInlinedUnaryOp(~, outputDims, fcnStr, MATLABOutputName, MATLABArgIdentifierNames)
    % Translate a node into code that directly inlines built-in functions. 

%   Copyright 2020-2023 The MathWorks, Inc.
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    result.Code = MATLABOutputName + " = " + "struct('value', " + fcnStr + "(" + MATLABArgIdentifierNames{1} + ".value)"  + ", 'rank', " + string(outputDims) + ");";
    result.NumOutputs = 1;
    result.Success = true; 
end 
