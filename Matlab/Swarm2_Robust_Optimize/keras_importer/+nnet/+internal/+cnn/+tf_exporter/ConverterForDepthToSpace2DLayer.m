classdef ConverterForDepthToSpace2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % See
    % matlab\toolbox\images\deep\+images\+depthToSpace\+internal\depthToSpaceForward.m
    % for the MATLAB implementation of CRD

    % Supported input formats: SSC[B], and SSCT
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            % Blocksize must be square for TF
            if this.Layer.BlockSize(1) ~= this.Layer.BlockSize(2)
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForDepthToSpace2DLayer", this.Layer.Name);
                warningNoBacktrace(this, msg);
                convertedLayer.WarningMessages(end+1) = msg;
                convertedLayer.Success = false;
                return;
            end
            block_size = this.Layer.BlockSize(1);
            switch this.Layer.Mode
                case "dcr"
                    % Use tf.nn.depth_to_space. It's is a function, not a
                    % layer, so we'll wrap it in a Lambda layer to make it
                    % one.
                    codeToMakeLayer = sprintf("DepthToSpaceLayer = layers.Lambda(lambda X: tf.nn.depth_to_space(X, %d))", block_size);
                case "crd"
                    % use our handwritten custom layer
                    codeToMakeLayer = sprintf("DepthToSpaceLayer = depthToSpaceCRDLayer(%d)", block_size);
                    convertedLayer.customLayerNames = "depthToSpaceCRDLayer";
                otherwise
                    convertedLayer.Success = false;
                    return
            end
            if this.layerAnalyzer.IsTemporal
                codeToCallLayer = sprintf("%s = layers.TimeDistributed(DepthToSpaceLayer)(%s)", this.OutputTensorName, this.InputTensorName);
            else
                codeToCallLayer = sprintf("%s = DepthToSpaceLayer(%s)", this.OutputTensorName, this.InputTensorName);
            end
            convertedLayer.layerCode = [codeToMakeLayer; codeToCallLayer];
        end
    end
end
