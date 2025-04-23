classdef ConverterForConvolution2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported DLT input formats:

    % For spatial conv: {SSC, SSCB, SSCT, SSCBT}. TF input formats are
    % always BSSC or BTSSC in these cases, and we export to TimeDistributed
    % conv2d.

    % For spatio-temporal conv: {SCT, SCBT}. TF input formats are always
    % BTSC in these cases, and we export to non-TimeDistributed conv2d.
    % But: In DLT the dimensions to conv over are SS, while in TF they are
    % TS. So we need to move the strides/padding for T to the front, and
    % permute the kernel.
    methods
        function convertedLayer = toTensorflow(this)
            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            Stride         = this.Layer.Stride;
            FilterSize     = this.Layer.FilterSize;
            DilationFactor = this.Layer.DilationFactor;
            PaddingSize    = this.Layer.PaddingSize;
            % Maybe handle spatio-temporal:
            isSpatioTemporal = ismember(this.InputFormat, ["SCT", "SCBT"]);
            if isSpatioTemporal
                % Convert hypers from SST to TSS ordering
                Stride         = Stride([2 1]);
                FilterSize     = FilterSize([2 1]);
                DilationFactor = DilationFactor([2 1]);
                PaddingSize    = PaddingSize(:,[3 4 1 2]);                  % PaddingSize is [t b l r]
            end
            % Assemble TF hyperparams
            pool_size     = this.Layer.NumFilters;
            kernel_size = sprintf("(%s)", join(string(FilterSize), ','));
            if all(Stride==1)
                strideStr = '';
            else
                strideStr = sprintf(", strides=(%s)", join(string(Stride), ','));
            end
            if isequal(this.Layer.PaddingMode, 'same')
                paddingStr = ', padding="same"';
            else
                % padding="valid". omit
                paddingStr = '';                                                % A padding layer may be added below.
            end
            if all(DilationFactor==1)
                dilationStr = '';
            else
                dilationStr = sprintf(", dilation_rate=(%s)", join(string(DilationFactor), ','));
            end
            if ~all(PaddingSize == 0) && ~isequal(this.Layer.PaddingValue, 0)
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForConvolutionLayers", this.Layer.Name, string(this.Layer.PaddingValue));
                warningNoBacktrace(this, msg);
                convertedLayer.WarningMessages(end+1) = msg;
            end

            % Generate code
            useTimeDistributed = this.layerAnalyzer.IsTemporal && ~isSpatioTemporal;
            if isequal(this.Layer.PaddingMode, 'same') || all(PaddingSize == 0)
                % Generate only a conv line
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, ...
                    "layers.Conv2D", "%d, %s%s%s%s", {pool_size, kernel_size, strideStr, paddingStr, dilationStr}, ...
                    layerName, useTimeDistributed);
            else
                % Generate a padding line and a conv line.
                t = PaddingSize(1);
                b = PaddingSize(2);
                l = PaddingSize(3);
                r = PaddingSize(4);
                padOutputTensorName = sprintf("%s_prepadded", this.OutputTensorName);
                convertedLayer.layerCode = [
                    kerasCodeLine(this, this.InputTensorName, padOutputTensorName, ...
                    "layers.ZeroPadding2D", "padding=((%d,%d),(%d,%d))", {t, b, l, r}, ...
                    "", useTimeDistributed)

                    kerasCodeLine(this, padOutputTensorName, this.OutputTensorName, ...
                    "layers.Conv2D", "%d, %s%s%s%s", {pool_size, kernel_size, strideStr, paddingStr, dilationStr}, ...
                    layerName, useTimeDistributed)
                    ];
            end

            % Format of kernel weights:
            % Keras and MATLAB are both: Height-Width-NumChannels-NumFilters: H x W x C x F
            % But we need to permute if isSpatioTemporal
            Weights = this.Layer.Weights;
            if isSpatioTemporal
                % Weights are HWCF. But W is really T. In TF, T needs to
                % become the first of the spatial dims. So we need to
                % permute to WHCF. That's [2 1 3 4]
                Weights = permute(Weights, [2 1 3 4]);
            end
            kerasWeights = permute(Weights, [4 3 2 1]);   % Switch memory ordering from Col-major to row-major.
            convertedLayer.LayerName = layerName;
            convertedLayer.weightNames    = ["kernel", "bias"];
            convertedLayer.weightArrays    = {kerasWeights; this.Layer.Bias};                                             % A matrix and a vector.
            convertedLayer.weightShapes    = {size(Weights, 1:4); numel(this.Layer.Bias)};  	% 4D and 1D.
        end
    end
end
