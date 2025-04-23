classdef ConverterForLeakyReLULayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: all
    methods
        function convertedLayer = toTensorflow(this)
            alpha = this.Layer.Scale;

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = layers.LeakyReLU(alpha=%f)(%s)", this.OutputTensorName, alpha, this.InputTensorName);
        end
    end
end