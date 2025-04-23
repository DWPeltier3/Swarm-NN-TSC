function result = translateTopKV2(this, node_def, MATLABOutputName, MATLABArgIdentifierNames, multiOutputNameMap)
%

%   Copyright 2022-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 

    outputNames = makeMultipleOutputArgs(this, MATLABOutputName, 2); 
    K = MATLABArgIdentifierNames(2);
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfTopKV2", outputNames, [MATLABArgIdentifierNames(1) K]);
    multiOutputNameMap(node_def.name) = {'values', 'indices'};
    
    result.Success = true;
    result.OpFunctions = "tfTopKV2";
end 
