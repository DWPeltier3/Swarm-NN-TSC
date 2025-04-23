classdef ConverterForClippedReLULayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: all
    methods
        function convertedLayer = toTensorflow(this)
            max_value = this.Layer.Ceiling;

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = layers.ReLU(max_value=%f)(%s)", this.OutputTensorName, max_value, this.InputTensorName);
        end
    end
end