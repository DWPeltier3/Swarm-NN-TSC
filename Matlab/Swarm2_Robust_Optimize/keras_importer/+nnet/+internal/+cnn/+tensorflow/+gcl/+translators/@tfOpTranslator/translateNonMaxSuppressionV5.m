function result = translateNonMaxSuppressionV5(this, node_def, numOutputs, MATLABOutputName, MATLABArgIdentifierNames, multiOutputNameMap)
%   Copyright 2021-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 

    outputNames = makeMultipleOutputArgs(this, MATLABOutputName, numOutputs); 
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfNonMaxSuppressionV5", outputNames, MATLABArgIdentifierNames);
    multiOutputNameMap(node_def.name) = {'selected_indices', 'selected_scores', 'valid_outputs'}; %#ok<NASGU>
    
    result.Success = true;
    result.OpFunctions = "tfNonMaxSuppressionV5";

end 
