classdef FlattenCStyleLayerTemplate < nnet.internal.app.plugins.DeferredPropertiesLayerTemplate
    % FlattenCStyleLayerTemplate  App support for nnet.keras.layer.FlattenCStyleLayer
    
    %   Copyright 2022-2023 The MathWorks, Inc.
    
    properties
        ConstructorName = "nnet.keras.layer.FlattenCStyleLayer"
        RequiredArguments = struct('Name', "flatten");
        OptionalArguments = [];
        Group = nnet.internal.app.plugins.layer.LayerGroup.Other;
    end
    
    methods
        function configureViewProperties(this)
            this.AppearsInPalette = false;
            this.LayerDisplayType = "FlattenCStyleLayer";
            this.LayerClassName = "nnet.keras.layer.FlattenCStyleLayer";
            this.SupportsUnlocking = false;
        end

        function tf = shouldLock(~, ~)
            % All TensorFlow layers are locked to prevent user edits.

            tf = true;
        end
    end
end