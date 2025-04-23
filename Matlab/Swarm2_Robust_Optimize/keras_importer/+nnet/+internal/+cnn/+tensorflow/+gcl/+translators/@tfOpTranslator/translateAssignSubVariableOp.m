function result = translateAssignSubVariableOp(~, MATLABOutputName, MATLABArgIdentifierNames)
    %   Copyright 2023 The MathWorks, Inc.
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 

    call = MATLABOutputName + " = " + "struct('value', " + MATLABArgIdentifierNames{1} + ", 'rank', ndims(" + MATLABArgIdentifierNames{1} + "));" + newline;
    result.Code = call + nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfSub", MATLABOutputName, {MATLABOutputName, MATLABArgIdentifierNames{2}});
    
    result.NumOutputs = 1;
    result.OpFunctions = "tfSub";
    result.Success = true; 
end