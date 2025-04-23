classdef ConverterForCrossChannelNormalizationLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSC[B][T]
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            alpha = this.Layer.Alpha;
            beta = this.Layer.Beta;
            bias = this.Layer.K;
            wcsize = this.Layer.WindowChannelSize;

            % alpha in TF is needs to be made smaller because TF doesn't
            % divide it by the window size, as matlab does.
            alpha = alpha/wcsize;

            % TF doesn't support even window size. Make the size odd if
            % it's even.
            if floor(wcsize/2) == wcsize/2
                newwcsize = max(3, wcsize-1);
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForCrossChannelNormalizationLayer",...
                    this.Layer.Name, wcsize, newwcsize);
                warningNoBacktrace(this, msg);
                convertedLayer.WarningMessages(end+1) = msg;
                wcsize = newwcsize;
            end
            depth_radius = (wcsize-1)/2;

            % tf.nn.local_response_normalization is a function, not a
            % layer, so we'll wrap it in a Lambda layer to make it one.
            % Then, unlike the other normalization layers, it needs to be
            % wrapped in TimeDistributed when applied to temporal input
            % tensors.
            codeToMakeLayer = sprintf("CCNormLayer = layers.Lambda(lambda X: tf.nn.local_response_normalization(X, depth_radius=%f, bias=%f, alpha=%f, beta=%f))",...
                depth_radius, bias, alpha, beta);
            if this.layerAnalyzer.IsTemporal
                codeToCallLayer = sprintf("%s = layers.TimeDistributed(CCNormLayer)(%s)", this.OutputTensorName, this.InputTensorName);
            else
                codeToCallLayer = sprintf("%s = CCNormLayer(%s)", this.OutputTensorName, this.InputTensorName);
            end
            convertedLayer.layerCode = [codeToMakeLayer; codeToCallLayer];
        end
    end
end
