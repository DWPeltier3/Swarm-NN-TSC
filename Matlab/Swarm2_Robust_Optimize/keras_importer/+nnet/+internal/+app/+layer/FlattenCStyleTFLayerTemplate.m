classdef FlattenCStyleTFLayerTemplate < nnet.internal.app.plugins.DeferredPropertiesLayerTemplate
    % FlattenCStyleTFLayerTemplate  App support for nnet.keras.layer.FlattenCStyleTFLayer
    
    %   Copyright 2022-2023 The MathWorks, Inc.
    
    properties
        ConstructorName = "nnet.keras.layer.FlattenCStyleTFLayer"
        RequiredArguments = struct('Name', "flatten");
        OptionalArguments = [];
        Group = nnet.internal.app.plugins.layer.LayerGroup.Other;
    end
    
    methods
        function configureViewProperties(this)
            this.AppearsInPalette = false;
            this.LayerDisplayType = "FlattenCStyleTFLayer";
            this.LayerClassName = "nnet.keras.layer.FlattenCStyleTFLayer";
            this.SupportsUnlocking = false;
        end

        function tf = shouldLock(~, ~)
            % All TensorFlow layers are locked to prevent user edits.

            tf = true;
        end
    end
end