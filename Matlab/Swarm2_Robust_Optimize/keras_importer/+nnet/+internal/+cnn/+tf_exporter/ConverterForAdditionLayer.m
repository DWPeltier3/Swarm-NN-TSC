classdef ConverterForAdditionLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.


    % MATLAB doc: " All inputs to an addition layer must have the same
    % dimension."

    % tf.keras doc: "It takes as input a list of tensors, all of the same
    % shape, and returns a single tensor (also of the same shape)." 
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = layers.Add()([%s])", this.OutputTensorName, join(this.InputTensorName, ', '));
        end
    end
end
