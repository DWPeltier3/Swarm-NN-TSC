classdef ConverterForLayerNormalizationLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: S*C[B][T]
    methods
        function convertedLayer = toTensorflow(this)
            % Generate code
            epsilon = this.Layer.Epsilon;
            inputFormat = this.InputFormat;
            spatialDims = find(char(inputFormat)=='S');
            numSpatialDims = numel(spatialDims);
            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            switch this.Layer.OperationDimension
                case 'auto'
                    if numSpatialDims > 1
                        convertedLayer = genLayernormCodeWithSpatialAxis(this, convertedLayer, layerName, epsilon, inputFormat);
                    else
                        convertedLayer = genLayernormCode(this, convertedLayer, layerName, epsilon);
                    end
                case 'channel-only'
                    convertedLayer = genLayernormCode(this, convertedLayer, layerName, epsilon);
                case 'spatial-channel'
                    if numSpatialDims > 0
                        convertedLayer = genLayernormCodeWithSpatialAxis(this, convertedLayer, layerName, epsilon, inputFormat);
                    else
                        convertedLayer = genLayernormCode(this, convertedLayer, layerName, epsilon);
                    end
                case 'batch-excluded'
                    convertedLayer = genLayernormCodeWithBatchExcluded(this, convertedLayer, layerName, epsilon);
                otherwise
                    convertedLayer.Success = false;
                    return
            end

            % Create TF weights: All 1D vectors. In TF they are have these
            % names and ordering: gamma, beta
            gamma = this.Layer.Scale;
            beta = this.Layer.Offset;
            convertedLayer.weightNames = ["gamma", "beta"];
            convertedLayer.weightArrays = {gamma, beta};
            convertedLayer.weightShapes = {numel(gamma), numel(beta)};
            convertedLayer.LayerName = layerName;
        end

        function convertedLayer = genLayernormCode(this, convertedLayer, layerName, epsilon)
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "layers.LayerNormalization", "axis=-1, epsilon=%f", {epsilon}, ...
                layerName, false);   % Normalization layers must NOT be time-distributed.
        end

        function convertedLayer = genLayernormCodeWithSpatialAxis(this, convertedLayer, layerName, epsilon, inputFormat)
            % TF's layers.LayerNormalization expects element-wise weights
            % when normalizing across the spatial and channel dimensions,
            % but layerNormalizationLayer does not support that currently.
            % Also, tfa.layers.GroupNormalization normalizes across the
            % time dimension, which is not the case with
            % layerNormalizationLayer. Use a custom layer.
            axisStart = nnet.internal.cnn.tf_exporter.FormatConverter.mlDimToTFDim(inputFormat, 1);
            axis = axisStart:numel(char(inputFormat))-1;
            axisStr = join(string(axis), ',');
            offsetShape = numel(this.Layer.Offset);
            scaleShape = numel(this.Layer.Scale);
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "LayerNormalizationLayer", "axis=(%s), epsilon=%f, offset_shape=%d, scale_shape=%d", {axisStr, epsilon, offsetShape, scaleShape}, ...
                layerName, false);    % Normalization layers must NOT be time-distributed.
                convertedLayer.customLayerNames = "LayerNormalizationLayer";
        end

        function convertedLayer = genLayernormCodeWithBatchExcluded(this, convertedLayer, layerName, epsilon)
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                "tfa.layers.GroupNormalization", "groups=1, axis=-1, epsilon=%f", {epsilon}, ...
                layerName, false);   % Normalization layers must NOT be time-distributed.
            convertedLayer.packagesNeeded = "tfa";
        end
    end
end
