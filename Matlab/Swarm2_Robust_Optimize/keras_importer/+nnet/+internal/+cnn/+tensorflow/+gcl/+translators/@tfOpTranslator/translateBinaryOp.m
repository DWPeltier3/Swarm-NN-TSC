function result = translateBinaryOp(~, tfOpFcnName, MATLABOutputName, MATLABArgIdentifierNames)
    % Translate a node into code that calls hand-written tfOp functions that 
    % expects two inputs.
    % If there are more than 2 inputs throw an assertion failure, as this
    % should never happen.

%   Copyright 2020-2023 The MathWorks, Inc.

    assert(numel(MATLABArgIdentifierNames) == 2);
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(tfOpFcnName, {MATLABOutputName}, MATLABArgIdentifierNames);
    result.NumOutputs = 1;
    result.OpFunctions = tfOpFcnName;
    result.Success = true; 
end 
