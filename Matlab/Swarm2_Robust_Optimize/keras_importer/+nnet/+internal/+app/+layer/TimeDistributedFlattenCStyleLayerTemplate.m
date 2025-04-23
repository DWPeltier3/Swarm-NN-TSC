classdef TimeDistributedFlattenCStyleLayerTemplate < nnet.internal.app.plugins.DeferredPropertiesLayerTemplate
    % TimeDistributedFlattenCStyleLayerTemplate  App support for nnet.keras.layer.TimeDistributedFlattenCStyleLayer

    %   Copyright 2022-2023 The MathWorks, Inc.
    
    properties
        ConstructorName = "nnet.keras.layer.TimeDistributedFlattenCStyleLayer" 
        RequiredArguments = struct("Name", "timedist");
        OptionalArguments = [];
        Group = nnet.internal.app.plugins.layer.LayerGroup.Other;
    end
    
    methods
        function configureViewProperties(this)
            this.AppearsInPalette = false;
            this.LayerDisplayType = "TimeDistributedFlattenCStyleLayer";
            this.LayerClassName = "nnet.keras.layer.TimeDistributedFlattenCStyleLayer";
            this.SupportsUnlocking = false;
        end

        function tf = shouldLock(~, ~)
            % All TensorFlow layers are locked to prevent user edits.

            tf = true;
        end
    end
end