function result = translateConv2D(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    
    inputTensorName = MATLABArgIdentifierNames{1}; 
    filtersTensorName = MATLABArgIdentifierNames{2}; 
    strides = node_def.attr.strides.list.i';
    padding = char(matlab.net.base64decode(node_def.attr.padding.s)); 
    if strcmp(padding,'EXPLICIT')
        explicit_padding = node_def.attr.explicit_paddings.list.i';
    else
        explicit_padding = {''};
    end
    if isfield(node_def.attr, 'dilations')
        dilations = node_def.attr.dilations.list.i';
    else
        dilations = {''};
    end
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfConv2D", {MATLABOutputName},[inputTensorName ...
                             filtersTensorName {['[' strjoin(strides) ']'],... 
                             ['"' padding '"'], ['[' strjoin(explicit_padding) ']'],...
                             ['[' strjoin(dilations) ']']}]); 

    result.OpFunctions = "tfConv2D";
    result.Success = true; 
end 
