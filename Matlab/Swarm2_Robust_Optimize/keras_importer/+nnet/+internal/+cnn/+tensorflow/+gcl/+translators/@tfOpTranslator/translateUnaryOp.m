function result = translateUnaryOp(~, tfOpFcnName, MATLABOutputName, MATLABArgIdentifierNames)
%   Translate a node into code that calls hand-written tfOp functions. 

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(tfOpFcnName, {MATLABOutputName}, MATLABArgIdentifierNames);
    result.NumOutputs = 1;
    result.OpFunctions = tfOpFcnName;
    result.Success = true; 
end
