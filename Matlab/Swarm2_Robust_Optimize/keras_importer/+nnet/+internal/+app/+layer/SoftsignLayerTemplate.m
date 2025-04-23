classdef SoftsignLayerTemplate < nnet.internal.app.plugins.DeferredPropertiesLayerTemplate
    % SoftsignLayerTemplate  App support for nnet.keras.layer.SoftsignLayer

    %   Copyright 2023 The MathWorks, Inc.

    properties
        ConstructorName = "nnet.keras.layer.SoftsignLayer"
        RequiredArguments = struct('Name', 'softsign');
        OptionalArguments = [];
        Group = nnet.internal.app.plugins.layer.LayerGroup.Other;
    end

    methods
        function configureViewProperties(this)
            this.AppearsInPalette = false;
            this.LayerDisplayType = "SoftsignLayer";
            this.LayerClassName = "nnet.keras.layer.SoftsignLayer";
            this.SupportsUnlocking = false;
        end

        function tf = shouldLock(~, ~)
            % All TensorFlow layers are locked to prevent user edits.

            tf = true;
        end
    end
end