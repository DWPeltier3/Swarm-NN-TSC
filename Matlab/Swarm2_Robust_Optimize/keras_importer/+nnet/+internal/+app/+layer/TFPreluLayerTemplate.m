classdef TFPreluLayerTemplate < nnet.internal.app.plugins.DeferredPropertiesLayerTemplate
    % TFPreluLayerTemplate  App support for nnet.keras.layer.PreluLayer
    %
    % This is the LayerTemplate for the Tensorflow version. Not to be
    % confused with app support for nnet.pytorch.layer.PReLULayer.
    %
    % nnet.keras.layer.PreluLayer is always "locked" by DND. This is
    % because it has 1 Trainable property (Alpha), and that property is
    % always set to be non-empty on construction (at least for realistic
    % usage). This means the layer always has non-empty Trainable
    % parameters, and therefore is always locked.
    
    %   Copyright 2022-2023 The MathWorks, Inc.
    
    properties
        ConstructorName = "nnet.keras.layer.PreluLayer" 
        RequiredArguments = struct("Name", "prelu", "Alpha", 0.1);
        OptionalArguments = [];
        Group = nnet.internal.app.plugins.layer.LayerGroup.Other;
    end
    
    methods
        function configureViewProperties(this)
            this.AppearsInPalette = false;
            this.LayerDisplayType = "PreluLayer";
            this.LayerClassName = "nnet.keras.layer.PreluLayer";
            this.SupportsUnlocking = false;
        end

        function tf = shouldLock(~, ~)
            % All TensorFlow layers are locked to prevent user edits.

            tf = true;
        end
    end
end