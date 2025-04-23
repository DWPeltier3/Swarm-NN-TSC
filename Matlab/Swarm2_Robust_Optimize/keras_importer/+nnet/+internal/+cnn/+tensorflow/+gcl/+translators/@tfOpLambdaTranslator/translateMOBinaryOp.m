function result = translateMOBinaryOp(~, tfOpFcnName, numOutputs, MATLABOutputName, MATLABArgIdentifierNames)
    % Translate an OpLambda function into code that calls a tfOp function that 
    % expects two inputs and multiple outputs.
    % If there are more than 2 inputs throw an assertion failure, as this
    % should never happen.

    %   Copyright 2023 The MathWorks, Inc.

    assert(numel(MATLABArgIdentifierNames) == 2);
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    MATLABOutputNames = cell(1, numOutputs);
    for i = 1:numOutputs
        MATLABOutputNames{i} = string(MATLABOutputName{1}) + "{" + num2str(i) + "}";
    end
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(tfOpFcnName, MATLABOutputNames, MATLABArgIdentifierNames);
    
    % The tfOpFunction is responsible for setting the rank
    result.ForwardRank = false;
    result.NumOutputs = numOutputs;
    
    result.OpFunctions = tfOpFcnName;
    result.Success = true; 
end 
