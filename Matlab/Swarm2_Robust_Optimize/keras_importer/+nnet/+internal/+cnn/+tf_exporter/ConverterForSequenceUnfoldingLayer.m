classdef ConverterForSequenceUnfoldingLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*C[B]
    % If there's no output T dimension, it's just the identity function on
    % the first input.
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if contains(this.OutputFormat, 'T')
                inSize = this.InputSize{1};
                inShape = "(" + join(string(inSize), ',') + ")";
                % Generate code
                convertedLayer.layerCode = sprintf("%s = SequenceUnfoldingLayer(%s)(%s, %s)", ...
                    this.OutputTensorName, inShape, this.InputTensorName(1), this.InputTensorName(2));
                convertedLayer.customLayerNames = "SequenceUnfoldingLayer";
            else
                convertedLayer.layerCode = sprintf("%s = %s", this.OutputTensorName, this.InputTensorName(1));
            end
        end
    end
end
