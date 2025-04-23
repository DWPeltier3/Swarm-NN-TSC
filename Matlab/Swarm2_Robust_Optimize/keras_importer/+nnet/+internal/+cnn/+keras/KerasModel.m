classdef KerasModel < handle
    
    % Copyright 2019-2021 The Mathworks, Inc.
    
    properties
        ClassName
        Config  % The parsed config. An array of KerasLayerInsideModel
        isTimeDistributed
        isTensorFlowModel = false;
    end
    
    methods(Abstract)
        % flatten() takes a KerasModel for an imported Keras Network,
        % flattens all KerasLayersInsideModel, and return correspoding layerSpecs. 
       [LayerSpecs, InputLayerIndices, OutputTensors] = flatten(this);
    end
    
end
        

