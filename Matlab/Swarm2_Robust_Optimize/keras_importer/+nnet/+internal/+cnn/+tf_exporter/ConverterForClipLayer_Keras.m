classdef ConverterForClipLayer_Keras < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: all
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = layers.Lambda(lambda X: tf.clip_by_value(X, clip_value_min=%f, clip_value_max=%f))(%s)", ...
                this.OutputTensorName, this.Layer.Min, this.Layer.Max, this.InputTensorName);
        end
    end
end