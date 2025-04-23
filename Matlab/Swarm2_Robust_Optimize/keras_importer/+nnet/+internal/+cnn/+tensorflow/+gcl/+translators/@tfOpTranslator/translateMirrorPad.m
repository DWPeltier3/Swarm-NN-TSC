function result = translateMirrorPad(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    if isfield(node_def.attr, 'mode')
        mode = "'" + string(char(matlab.net.base64decode(node_def.attr.mode.s))) + "'";
    end 
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfMirrorPad", MATLABOutputName, [MATLABArgIdentifierNames {mode}]); 
    result.OpFunctions = "tfMirrorPad";
    result.Success = true; 
end 
