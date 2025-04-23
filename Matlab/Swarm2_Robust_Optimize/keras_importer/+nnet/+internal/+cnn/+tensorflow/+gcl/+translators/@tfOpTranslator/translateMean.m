function result = translateMean(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    keep_dims = "false";
    if isfield(node_def.attr, 'keep_dims')
        if node_def.attr.keep_dims.b
            keep_dims = "true";
        end
    end

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfMean", MATLABOutputName, [MATLABArgIdentifierNames keep_dims]); 
    result.Success = true;
    result.OpFunctions = "tfMean";
end 
