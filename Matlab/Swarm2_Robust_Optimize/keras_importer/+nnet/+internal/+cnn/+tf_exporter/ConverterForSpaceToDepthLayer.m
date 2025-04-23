classdef ConverterForSpaceToDepthLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSC[B], and SSCT
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            % Blocksize must be square for TF
            if this.Layer.BlockSize(1) ~= this.Layer.BlockSize(2)
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForSpaceToDepthLayer", this.Layer.Name);
                warningNoBacktrace(this, msg);
                convertedLayer.WarningMessages(end+1) = msg;
                convertedLayer.Success = false;
                return;

            end
            block_size = this.Layer.BlockSize(1);

            % tf.nn.space_to_depth is a function, not a layer, so we'll
            % wrap it in a Lambda layer to make it one. Then, it needs to
            % be wrapped in TimeDistributed when applied to temporal input
            % tensors.
            codeToMakeLayer = sprintf("SpaceToDepthLayer = layers.Lambda(lambda X: tf.nn.space_to_depth(X, %d))", block_size);
            if this.layerAnalyzer.IsTemporal
                codeToCallLayer = sprintf("%s = layers.TimeDistributed(SpaceToDepthLayer)(%s)", this.OutputTensorName, this.InputTensorName);
            else
                codeToCallLayer = sprintf("%s = SpaceToDepthLayer(%s)", this.OutputTensorName, this.InputTensorName);
            end
            convertedLayer.layerCode = [codeToMakeLayer; codeToCallLayer];

        end
    end
end
