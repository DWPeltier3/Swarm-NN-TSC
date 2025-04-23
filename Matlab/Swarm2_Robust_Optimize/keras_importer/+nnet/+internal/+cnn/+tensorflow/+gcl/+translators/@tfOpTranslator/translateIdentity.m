function result = translateIdentity(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;
    if ~isempty(node_def.attr.x_output_shapes.list.shape.dim)
        variableNdims = numel(node_def.attr.x_output_shapes.list.shape.dim);
        result.Code = MATLABOutputName + " = " + "struct('value', " + MATLABArgIdentifierNames{1} + '.value'  + ", 'rank', " + string(variableNdims) + ");";
    else
        result.Code = MATLABOutputName + ".value" + " = " + MATLABArgIdentifierNames{1} + ".value" + ";";
        result.ForwardRank = true;
    end
    result.Success = true;
end 
