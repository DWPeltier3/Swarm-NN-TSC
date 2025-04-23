classdef BinaryCrossEntropyRegressionLayerTemplate < ...
        nnet.internal.app.plugins.DeferredPropertiesLayerTemplate & ...
        nnet.internal.app.plugins.layer.CustomizedExport
    % BinaryCrossEntropyRegressionLayerTemplate  App support for nnet.keras.layer.BinaryCrossEntropyRegressionLayer

    %   Copyright 2023 The MathWorks, Inc.
    
    properties
        ConstructorName = "nnet.keras.layer.BinaryCrossEntropyRegressionLayer" 
        RequiredArguments = struct("Name", "bce", "isRnn", true);
        OptionalArguments = [];
        Group = nnet.internal.app.plugins.layer.LayerGroup.Other;
    end
    
    methods
        function configureViewProperties(this)
            this.AppearsInPalette = false;
            this.LayerDisplayType = "BinaryCrossEntropyRegressionLayer";
            this.LayerClassName = "nnet.keras.layer.BinaryCrossEntropyRegressionLayer";
            this.SupportsUnlocking = false;
        end

        function configureLayerProperties(this)

            % Hide Description, Type and ResponseNames properties
            this.LayerProperties.Description.Visible = false;
            this.LayerProperties.Type.Visible = false;
            this.LayerProperties.ResponseNames.Visible = false;

            % BatchIdx can only be values 2 or 4.
            this.LayerProperties.BatchIdx.Widget = deepapp.internal.plugins.layer.display.EnumWidget(["2", "4"]);
        end

        function configureOutputProperties(this)
            % This is an output layer, so we need to configure its output
            % ports correctly.
            
            this.IsOutputLayer = true;

            % Output layers have no output ports
            this.PrototypeOutputNames = {};
        end

        function layer = createLayer(~, dndProperties, ~)

            % Map from the layer's BatchIdx property to the isRnn ctor arg
            isRnn = (dndProperties.BatchIdx == 2);

            % Construct layer
            layer = nnet.keras.layer.BinaryCrossEntropyRegressionLayer( ...
                dndProperties.Name, isRnn);
        end

        function [fcnName, fcnInputs] = generateLayerCode(~, exportedLayer, ~)

            % Map from the layer's BatchIdx property to the isRnn ctor arg
            isRnn = (exportedLayer.BatchIdx == 2);

            fcnName = "nnet.keras.layer.BinaryCrossEntropyRegressionLayer";
            fcnInputs = {exportedLayer.Name, isRnn};
        end

        function tf = shouldLock(~, ~)
            % All TensorFlow layers are locked to prevent user edits.

            tf = true;
        end
    end
end