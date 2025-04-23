classdef ConverterForPreluLayer_Keras < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: all.
    % The keras importer custom layer only handles scalar alpha. To
    % export that, we need to set shared_axes to 1:rank-1 so the scalar
    % is shared across all TS*C dimensions in TF.

    methods
        function convertedLayer = toTensorflow(this)
            layerName = this.Layer.Name + "_";
            rankWithoutB = strlength(erase(this.InputFormat, "B"));  % S*C[B]T
            shared_axes = "[" + join(string(1:rankWithoutB), ", ") + "]";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = layers.PReLU(name=""%s"", shared_axes=%s)(%s)", ...
                this.OutputTensorName, layerName, shared_axes, this.InputTensorName);
            % Set the weights
            convertedLayer.LayerName = layerName;             % Name it, because it has weights.
            convertedLayer.weightNames = "alpha";
            convertedLayer.weightShapes = {ones(1, rankWithoutB)};  % E.g., [1 1 1] for SSCB.
            convertedLayer.weightArrays = {this.Layer.Alpha};
        end
    end
end