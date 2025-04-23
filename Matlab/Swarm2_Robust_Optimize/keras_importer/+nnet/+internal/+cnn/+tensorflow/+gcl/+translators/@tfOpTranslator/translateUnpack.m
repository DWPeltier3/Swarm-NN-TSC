function result = translateUnpack(this, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2022-2023 The MathWorks, Inc.
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    num = str2double(node_def.attr.num.i); 
    axis = 0;
    if isfield(node_def.attr, 'axis')
        axis = node_def.attr.axis.i; 
    end 
    
    if num > 1
        outputNames = makeMultipleOutputArgs(this, MATLABOutputName, (num));
    else 
        outputNames = {MATLABOutputName};
    end
    
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfUnpack", ...
        outputNames, [MATLABArgIdentifierNames {num, axis}]);

    if num > 1
        result.NumOutputs = num; 
    end
    result.OpFunctions = "tfUnpack";
    result.Success = true; 
end
