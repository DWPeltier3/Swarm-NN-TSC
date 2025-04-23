function result = translateReadVariableOp(this, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    variableNdims = numel(node_def.attr.x_output_shapes.list.shape.dim);
    result.Code = MATLABOutputName + " = " + "struct('value', " + MATLABArgIdentifierNames{1}  + ", 'rank', " + string(variableNdims) + ");";
    result.NumOutputs = 1; 
    result.Success = true; 
end
