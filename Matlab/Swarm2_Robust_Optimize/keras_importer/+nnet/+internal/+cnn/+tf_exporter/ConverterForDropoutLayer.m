classdef ConverterForDropoutLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: all
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = layers.Dropout(%f)(%s)", this.OutputTensorName, this.Layer.Probability, this.InputTensorName);
        end
    end
end
