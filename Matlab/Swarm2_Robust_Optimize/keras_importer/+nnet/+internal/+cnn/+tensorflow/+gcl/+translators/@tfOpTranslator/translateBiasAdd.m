function result = translateBiasAdd(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2022-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 

    if isfield(node_def.attr, 'data_format') && isfield(node_def.attr.data_format, 's')
        dataFormat = char(matlab.net.base64decode(node_def.attr.data_format.s));
    else
        dataFormat = 'NHWC';
    end
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfBiasAdd", {MATLABOutputName}, [MATLABArgIdentifierNames ['"' dataFormat '"']]);
    result.NumOutputs = 1;
    result.Success = true;
    result.OpFunctions = "tfBiasAdd";
end 
