classdef ConverterForFlattenLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*C[B][T]
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            permuteLine = string.empty;
            flattenInputName = this.InputTensorName;
            dataNumDims = numel(this.InputSize{1});
            if dataNumDims > 1
                % We need to permute first so we get column-major
                % flattening as in MATLAB
                flattenInputName = this.InputTensorName+"perm";
                permuteLine = kerasCodeLine(this, this.InputTensorName, flattenInputName, ...
                    "layers.Permute", "(%s)", {join(string(dataNumDims:-1:1), ',')}, ...
                    "", this.layerAnalyzer.IsTemporal);
            end
            convertedLayer.layerCode = [
                permuteLine
                kerasCodeLine(this, flattenInputName, this.OutputTensorName, ...
                "layers.Flatten", "", {}, ...
                "", this.layerAnalyzer.IsTemporal)
                ];
        end
    end
end