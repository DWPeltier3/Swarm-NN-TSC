function result = translateUnsupportedOp(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames)
    % This translator is intended for translating unsupported operators.
    % an error will be generated to ensure that this layer doesn't execute 
    % successfully.  

%   Copyright 2020-2023 The MathWorks, Inc.
    
    if numOutputs > 1
        outputNames = makeMultipleOutputArgs(this, MATLABOutputName, numOutputs); 
    else
        outputNames = {MATLABOutputName}; 
    end
    
    % Write the call for this function
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tf"+node_def.op, outputNames, MATLABArgIdentifierNames);
    result.OpFunctions = "tf"+node_def.op;
    result.IsCommenting = true; 
    result.Comment = "% Placeholder function for " + node_def.op; 
    result.Node = node_def; 
    result.Success = false; 
end
