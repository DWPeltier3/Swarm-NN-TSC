function result = translateSqueeze(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    if isfield(node_def.attr, 'squeeze_dims')
        axis = node_def.attr.squeeze_dims.list.i';
    else
        axis = {};
    end
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfSqueeze", {MATLABOutputName}, [MATLABArgIdentifierNames, {['[' strjoin(axis) ']']}]); 
    result.OpFunctions = "tfSqueeze"; 
    result.Success = true; 
end 
