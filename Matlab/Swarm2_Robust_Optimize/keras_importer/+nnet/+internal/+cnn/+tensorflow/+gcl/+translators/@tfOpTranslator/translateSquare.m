function result = translateSquare(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    fcnStr = "power";
    variableNdims = numel(node_def.attr.x_output_shapes.list.shape.dim);
    result.Code = MATLABOutputName + " = " + "struct('value', " + fcnStr + "(" + MATLABArgIdentifierNames{1} + ".value, " + "2), 'rank', " + string(variableNdims) + ");";
    result.Success = true; 
end 
