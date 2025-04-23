function result = translateAvgPool(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    ksize = node_def.attr.ksize.list.i';
    strides = node_def.attr.strides.list.i';
    padding = char(matlab.net.base64decode(node_def.attr.padding.s));
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfAvgPool", {MATLABOutputName}, ...
        [MATLABArgIdentifierNames {['[' strjoin(ksize) ']'], ...
        ['[' strjoin(strides) ']'], ['"' padding '"']}]); 
    
    result.OpFunctions = "tfAvgPool";
    result.Success = true; 
end 
