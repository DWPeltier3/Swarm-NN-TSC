function result = translateResizeBilinear(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

% Copyright 2021-2023 The MathWorks, Inc.
    
    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    
    half_pixel_centers = "'asymmetric'"; 
    if isfield(node_def.attr, 'half_pixel_centers')
        if node_def.attr.half_pixel_centers.b
            half_pixel_centers = "'half-pixel'";
        end 
    end 
    
    % Add DLT labels to the input to dlresize
    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfResize", {MATLABOutputName}, [{MATLABArgIdentifierNames(1)} ...
        {MATLABArgIdentifierNames{2}}, {half_pixel_centers}, {"'linear'"}]);

    result.OpFunctions = "tfResize";
    result.Success = true;
    result.NumOutputs = 1; 
end 
