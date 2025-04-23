function result = translateDepthToSpace(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    block_size = node_def.attr.block_size.i;
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfDepthToSpace", {MATLABOutputName}, {MATLABArgIdentifierNames{1}, block_size});    
    result.OpFunctions = "tfDepthToSpace"; 
    result.Success = true; 
end 
