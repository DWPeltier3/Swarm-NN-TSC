classdef ConverterForClassificationOutputLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: 1*CB[T]
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            inSize = this.InputSize{1};
            if numel(inSize)>1
                % DLT input is 1*CB[T], and gets flattened to CB[T]. TF input
                % is B[T]1*C and gets flattened to B[T]C.
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.Flatten",...
                    "", {},...
                    "", this.layerAnalyzer.IsTemporal);
            else
                % CB[T] --> CB[T]. Generate code to set the output tensor
                % equal to the input tensor
                convertedLayer.layerCode = sprintf("%s = %s", this.OutputTensorName, this.InputTensorName);
            end
        end
    end
end
