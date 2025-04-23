function result = translateMaxPool(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    ksize = node_def.attr.ksize.list.i';
    strides = node_def.attr.strides.list.i';
    padding = char(matlab.net.base64decode(node_def.attr.padding.s));
    if strcmp(padding,'EXPLICIT')
        explicit_padding = node_def.attr.explicit_paddings.list.i';
    else
        explicit_padding = {''};
    end
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfMaxPool", {MATLABOutputName}, ...
        [MATLABArgIdentifierNames {['[' strjoin(ksize) ']'], ...
        ['[' strjoin(strides) ']'], ['"' padding '"'], ['[' strjoin(explicit_padding) ']']}]); 
    
    result.OpFunctions = "tfMaxPool";
    result.Success = true; 
end 
