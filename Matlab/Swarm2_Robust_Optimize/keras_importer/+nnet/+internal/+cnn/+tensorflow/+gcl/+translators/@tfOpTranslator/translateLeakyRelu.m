function result = translateLeakyRelu(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 

    if isfield(node_def.attr, 'alpha')
        alpha = node_def.attr.alpha.f;
    else 
        alpha = 0.2; 
    end 
    
    % We expect alpha to always be scalar, throw an assertion failure if that
    % is not the case
    assert(isscalar(alpha)); 
    alpha = num2str(alpha); 

    fcnName = "leakyrelu";
    variableNdims = numel(node_def.attr.x_output_shapes.list.shape.dim);
    result.Code = MATLABOutputName + " = " + "struct('value', " + fcnName + "(" + MATLABArgIdentifierNames{1} + ".value, " + alpha + "), 'rank', " + string(variableNdims) + ");";
    result.Success = true;
end 
