classdef ConverterForAveragePooling3DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported DLT input formats:

    % For spatial: {SSSC, SSSCB, SSSCT, SSSCBT}. TF input formats are
    % always BSSSC or BTSSSC in these cases, and we export to
    % TimeDistributed pooling.

    % For spatio-temporal: {SSCT, SSCBT}. TF input formats are always BTSSC
    % in these cases, and we export to non-TimeDistributed pooling. But: In
    % DLT the dimensions to pool over are SST, while in TF they are TSS. So
    % we need to move the strides/padding for T to the front.
    methods
        function convertedLayer = toTensorflow(this)
            Stride         = this.Layer.Stride;
            PoolSize       = this.Layer.PoolSize;
            PaddingSize    = this.Layer.PaddingSize;
            % Maybe handle spatio-temporal:
            isSpatioTemporal = ismember(this.InputFormat, ["SSCT", "SSCBT"]);
            if isSpatioTemporal
                % Convert hypers from SST to TSS ordering
                Stride         = Stride([3 1 2]);
                PoolSize       = PoolSize([3 1 2]);
                PaddingSize    = PaddingSize(:,[3 1 2]);                    % PaddingSize is a 2-by-3 matrix [t l f;b r k]
            end
            % Assemble TF hyperparams
            pool_size	= sprintf("(%s)", join(string(PoolSize), ','));     	% (H,W,D)
            strides     = sprintf("(%s)", join(string(Stride), ','));        	% (H,W,D)

            % Generate code
            useTimeDistributed = this.layerAnalyzer.IsTemporal && ~isSpatioTemporal;
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if isequal(this.Layer.PaddingMode, 'same') && isequal(this.Layer.PaddingValue, "mean")
                % Generate only a avgpool line with same padding, because
                % Keras uses mean padding when you specify 'same'.
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.AveragePooling3D",...
                    "pool_size=%s, strides=%s, padding=""same""", {pool_size, strides},...
                    "", useTimeDistributed);
            elseif all(PaddingSize(:) == 0)
                % Generate only a avgpool line with valid padding
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.AveragePooling3D",...
                    "pool_size=%s, strides=%s", {pool_size, strides},...
                    "", useTimeDistributed);
            else
                if ~isequal(this.Layer.PaddingValue, 0)
                    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForAveragePoolingLayers", this.Layer.Name, string(this.Layer.PaddingValue));
                    warningNoBacktrace(this, msg);
                    convertedLayer.WarningMessages(end+1) = msg;
                end
                % This handles manual AND SAME padding. First pad with
                % zeros, then pool with valid padding. 3D padding layout is
                % a 2-by-3 matrix [t l f;b r k]
                t = PaddingSize(1,1);
                b = PaddingSize(2,1);
                l = PaddingSize(1,2);
                r = PaddingSize(2,2);
                f = PaddingSize(1,3);
                k = PaddingSize(2,3);
                padOutputTensorName = sprintf("%s_prepadded", this.OutputTensorName);
                convertedLayer.layerCode = [
                    kerasCodeLine(this, this.InputTensorName, padOutputTensorName, "layers.ZeroPadding3D",...
                    "padding=((%d,%d),(%d,%d),(%d,%d))", {t, b, l, r, f, k},...
                    "", useTimeDistributed)

                    kerasCodeLine(this, padOutputTensorName, this.OutputTensorName, "layers.AveragePooling3D",...
                    "pool_size=%s, strides=%s", {pool_size, strides},...
                    "", useTimeDistributed)
                    ];
            end
        end
    end
end
