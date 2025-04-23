classdef ConverterForGELULayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: all
    methods
        function convertedLayer = toTensorflow(this)
            approximation = this.Layer.Approximation;

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if isequal(approximation, 'none')
                convertedLayer.layerCode = sprintf("%s = layers.Activation('gelu')(%s)", this.OutputTensorName, this.InputTensorName);
            else
                approximate = 'True';
                convertedLayer.layerCode = sprintf("%s = layers.Activation(lambda x : tf.nn.gelu(x, approximate=%s))(%s)", this.OutputTensorName, approximate, this.InputTensorName);
            end
        end
    end
end