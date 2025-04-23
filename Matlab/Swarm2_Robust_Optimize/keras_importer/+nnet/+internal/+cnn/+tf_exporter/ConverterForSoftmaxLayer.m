classdef ConverterForSoftmaxLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*C[B][T]
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            % DLT softmax operates over the C dim, which must always be
            % present. All supported TF formats end in C, which is axis=1,
            % which is the default. So a simple call to the default Softmax
            % is all we need.
            convertedLayer.layerCode = sprintf("%s = layers.Softmax()(%s)", this.OutputTensorName, this.InputTensorName);
        end
    end
end