classdef ZeroPadding1dLayerTemplate < ...
        nnet.internal.app.plugins.DeferredPropertiesLayerTemplate & ...
        nnet.internal.app.plugins.layer.CustomizedExport
    % ZeroPadding1dLayerTemplate  App support for nnet.keras.layer.ZeroPadding1dLayer

    %   Copyright 2023 The MathWorks, Inc.
    
    properties
        ConstructorName = "nnet.keras.layer.ZeroPadding1dLayer" 
        RequiredArguments = struct("Name", "zeropad-1d", "Amounts", 1);
        OptionalArguments = [];
        Group = nnet.internal.app.plugins.layer.LayerGroup.Other;
    end
    
    methods
        function configureViewProperties(this)
            this.AppearsInPalette = false;
            this.LayerDisplayType = "ZeroPadding1dLayer";
            this.LayerClassName = "nnet.keras.layer.ZeroPadding1dLayer";
            this.SupportsUnlocking = false;
        end

        function layer = createLayer(~, dndProperties, ~)

            % Map from the layer's leftPad/rightPad properties to the
            % Amounts ctor arg
            amounts = [dndProperties.leftPad, dndProperties.rightPad];

            % Construct layer
            layer = nnet.keras.layer.ZeroPadding1dLayer( ...
                dndProperties.Name, amounts);
        end


        function [fcnName, fcnInputs] = generateLayerCode(~, exportedLayer, ~)
            
            fcnName = "nnet.keras.layer.ZeroPadding1dLayer";

            % Map from the layer's leftPad/rightPad properties to the
            % Amounts ctor arg
            amounts = [exportedLayer.leftPad, exportedLayer.rightPad];
            fcnInputs = {exportedLayer.Name, amounts};
        end

        function tf = shouldLock(~, ~)
            % All TensorFlow layers are locked to prevent user edits.

            tf = true;
        end
    end
end