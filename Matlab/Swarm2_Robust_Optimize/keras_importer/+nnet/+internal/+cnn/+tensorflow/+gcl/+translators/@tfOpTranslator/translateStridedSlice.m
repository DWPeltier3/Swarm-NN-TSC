function result = translateStridedSlice(~, node_def, MATLABOutputName, MATLABArgIdentifierNames)
%

%   Copyright 2020-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult; 
    
    if isfield(node_def.attr, 'begin_mask')
        beginMask = node_def.attr.begin_mask.i;
    else
        beginMask = "[]"; 
    end

    if isfield(node_def.attr, 'end_mask')
        endMask = node_def.attr.end_mask.i;
    else 
        endMask = "[]"; 
    end 

    if isfield(node_def.attr, 'ellipsis_mask')
        ellipsisMask = node_def.attr.ellipsis_mask.i; 
    else 
        ellipsisMask = "[]"; 
    end

    if isfield(node_def.attr, 'new_axis_mask')
        newAxisMask = node_def.attr.new_axis_mask.i; 
    else 
        newAxisMask = "[]"; 
    end     

    if isfield(node_def.attr, 'shrink_axis_mask')
        shrinkMask = node_def.attr.shrink_axis_mask.i; 
    else 
        shrinkMask = "[]"; 
    end 

    result.Code = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("tfStridedSlice", {MATLABOutputName}, [MATLABArgIdentifierNames {beginMask, endMask, ellipsisMask, newAxisMask, shrinkMask}]);

    result.OpFunctions = ["tfStridedSlice" "tfSqueeze"]; 
    result.Success = true; 
end 
