classdef ConverterForRegressionOutputLayer <nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supports 2D and 3D image input tensors
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            % Maybe insert a Flatten layer. RegressionOutputLayer
            % flattens 1*CB[T] to CB[T].
            inSize = this.InputSize{1};    % Excludes B[T] dimensions
            if ~this.layerAnalyzer.IsTemporal && numel(inSize)>1 && all(inSize(1:end-1)==1)
                % Input is 1*CB. Generate a flatten layer.
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.Flatten",...
                    "", {},...
                    "", false);
            else
                % Input is either CB[T] or S+CB[T]. When T is present, this
                % layer DOES NOT flatten the output! Generate code to set
                % the output tensor equal to the input tensor
                convertedLayer.layerCode = sprintf("%s = %s", this.OutputTensorName, this.InputTensorName);
            end
        end
    end
end
