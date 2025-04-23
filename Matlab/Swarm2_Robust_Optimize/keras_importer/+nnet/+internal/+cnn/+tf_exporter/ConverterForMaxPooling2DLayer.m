classdef ConverterForMaxPooling2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported DLT input formats:

    % For spatial: {SSC, SSCB, SSCT, SSCBT}. TF input formats are always
    % BSSC or BTSSC in these cases, and we export to TimeDistributed
    % pooling.

    % For spatio-temporal: {SCT, SCBT}. TF input formats are always BTSC in
    % these cases, and we export to non-TimeDistributed pooling. But: In
    % DLT the dimensions to pool over are SS, while in TF they are TS. So
    % we need to move the strides/padding for T to the front.

    % ATTEMPTS TO SUPPORT UNPOOLING OUTPUTS. BUT TENSORFLOW DOESN'T
    %  BEHAVE THE SAME. IT PRODUCES DIFFERENT SIZED TENSORS THAN MATLAB.

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            if this.Layer.HasUnpoolingOutputs
                convertedLayer.Success = false;
                return
%                 error(message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForMaxPooling2DLayer", this.Layer.Name));
            end

            Stride         = this.Layer.Stride;
            PoolSize       = this.Layer.PoolSize;
            PaddingSize    = this.Layer.PaddingSize;
            % Maybe handle spatio-temporal:
            isSpatioTemporal = ismember(this.InputFormat, ["SCT", "SCBT"]);
            if isSpatioTemporal
                % Convert hypers from SST to TSS ordering
                Stride         = Stride([2 1]);
                PoolSize       = PoolSize([2 1]);
                PaddingSize    = PaddingSize(:,[3 4 1 2]);                  % PaddingSize is [t b l r]
            end
            % Assemble TF hyperparams
            pool_size	= sprintf("(%s)", join(string(PoolSize), ','));     	% (H,W)
            strides     = sprintf("(%s)", join(string(Stride), ','));        % (H,W)

            % Generate code
            useTimeDistributed = this.layerAnalyzer.IsTemporal && ~isSpatioTemporal;
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if isequal(this.Layer.PaddingMode, 'same')
                % Generate only a maxpool line with same padding.
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.MaxPool2D", "pool_size=%s, strides=%s, padding=""same""", {pool_size, strides},...
                    "", useTimeDistributed);
            elseif all(PaddingSize == 0)
                % Generate only a maxpool line with valid padding

                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.MaxPool2D", "pool_size=%s, strides=%s", {pool_size, strides},...
                    "", useTimeDistributed);
            else
                % This handles nonzero manual padding.
                % First pad with zeros, then pool with valid padding.
                t = PaddingSize(1);
                b = PaddingSize(2);
                l = PaddingSize(3);
                r = PaddingSize(4);
                padOutputTensorName = sprintf("%s_prepadded", this.OutputTensorName(1));
                line1 = kerasCodeLine(this, this.InputTensorName, padOutputTensorName,...
                    "layers.ZeroPadding2D", "padding=((%d,%d),(%d,%d))", {t, b, l, r},...
                    "", useTimeDistributed);

                line2 = kerasCodeLine(this, padOutputTensorName, this.OutputTensorName,...
                    "layers.MaxPool2D", "pool_size=%s, strides=%s", {pool_size, strides},...
                    "", useTimeDistributed);
                convertedLayer.layerCode = [line1; line2];
            end
        end
    end
end

