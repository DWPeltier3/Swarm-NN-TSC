classdef ClipLayerTemplate < nnet.internal.app.plugins.DeferredPropertiesLayerTemplate
    % ClipLayerTemplate  App support for nnet.keras.layer.ClipLayer

    %   Copyright 2022-2023 The MathWorks, Inc.

    properties
        ConstructorName = "nnet.keras.layer.ClipLayer"
        RequiredArguments = struct('Name', 'clip', "Min", -inf, "Max", inf);
        OptionalArguments = [];
        Group = nnet.internal.app.plugins.layer.LayerGroup.Other;
    end

    methods
        function configureLayerProperties(this)
            this.LayerProperties.Min.Widget = iNumericVectorWidget();
            this.LayerProperties.Min.DisplayFormatter = iNumericVectorDisplayFormatter();

            this.LayerProperties.Max.Widget = iNumericVectorWidget();
            this.LayerProperties.Max.DisplayFormatter = iNumericVectorDisplayFormatter();
        end

        function configureViewProperties(this)
            this.AppearsInPalette = false;
            this.LayerDisplayType = "ClipLayer";
            this.LayerClassName = "nnet.keras.layer.ClipLayer";
            this.SupportsUnlocking = false;
        end

        function tf = shouldLock(~, ~)
            % All TensorFlow layers are locked to prevent user edits.

            tf = true;
        end
    end
end

% helpers
function widget = iNumericVectorWidget()
widget = deepapp.internal.plugins.layer.display.NumericVectorWidget();
end

function displayFormatter = iNumericVectorDisplayFormatter()
displayFormatter = deepapp.internal.plugins.layer.display.NumericVectorDisplayFormatter();
end