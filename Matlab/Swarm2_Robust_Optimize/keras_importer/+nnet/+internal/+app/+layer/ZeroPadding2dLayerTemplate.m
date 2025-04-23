classdef ZeroPadding2dLayerTemplate < ...
        nnet.internal.app.plugins.DeferredPropertiesLayerTemplate & ...
        nnet.internal.app.plugins.layer.CustomizedExport
    % ZeroPadding2dLayerTemplate  App support for nnet.keras.layer.ZeroPadding2dLayer

    %   Copyright 2023 The MathWorks, Inc.
    
    properties
        ConstructorName = "nnet.keras.layer.ZeroPadding2dLayer" 
        RequiredArguments = struct("Name", "zeropad-2d", "Amounts", 1);
        OptionalArguments = [];
        Group = nnet.internal.app.plugins.layer.LayerGroup.Other;
    end
    
    methods
        function configureViewProperties(this)
            this.AppearsInPalette = false;
            this.LayerDisplayType = "ZeroPadding2dLayer";
            this.LayerClassName = "nnet.keras.layer.ZeroPadding2dLayer";
            this.SupportsUnlocking = false;
        end

        function layer = createLayer(~, dndProperties, ~)

            % Map from the layer's Top/Bottom/Left/Right properties to the
            % Amounts ctor arg
            amounts = [dndProperties.Top, dndProperties.Bottom, ...
                dndProperties.Left, dndProperties.Right];

            % Construct layer
            layer = nnet.keras.layer.ZeroPadding2dLayer( ...
                dndProperties.Name, amounts);
        end

        function [fcnName, fcnInputs] = generateLayerCode(~, exportedLayer, ~)

            fcnName = "nnet.keras.layer.ZeroPadding2dLayer";

            % Map from the layer's Top/Bottom/Left/Right properties to the
            % Amounts ctor arg
            amounts = [exportedLayer.Top, exportedLayer.Bottom,...
                exportedLayer.Left, exportedLayer.Right];

            fcnInputs = {exportedLayer.Name, amounts};
        end

        function tf = shouldLock(~, ~)
            % All TensorFlow layers are locked to prevent user edits.

            tf = true;
        end
    end
end