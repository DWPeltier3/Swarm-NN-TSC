classdef ConverterForConcatenationLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*C[B][T]
    methods
        function convertedLayer = toTensorflow(this)
            dltFormat = this.InputFormat(1);
            dltDim = this.Layer.Dim;
            tfDim = nnet.internal.cnn.tf_exporter.FormatConverter.mlDimToTFDim(dltFormat, dltDim);

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = layers.Concatenate(axis=%d)([%s])", ...
                this.OutputTensorName, tfDim, join(this.InputTensorName, ', '));
        end
    end
end
