classdef ConverterForSequenceFoldingLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*C[B][T]
    % If there's no T dimension, it's just the identity function.
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if contains(this.InputFormat, 'T')
                inSize = this.InputSize{1};
                inShape = "(" + join(string(inSize), ',') + ")";
                % Generate code
                convertedLayer.layerCode = sprintf("%s,%s = SequenceFoldingLayer(%s)(%s)", ...
                    this.OutputTensorName(1), this.OutputTensorName(2), inShape, this.InputTensorName);
                convertedLayer.customLayerNames = "SequenceFoldingLayer";
            else
                convertedLayer.layerCode = sprintf("%s = %s", this.OutputTensorName(1), this.InputTensorName);
            end
        end
    end
end
