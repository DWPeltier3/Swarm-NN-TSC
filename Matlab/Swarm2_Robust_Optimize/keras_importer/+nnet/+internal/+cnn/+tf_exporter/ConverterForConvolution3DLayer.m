classdef ConverterForConvolution3DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported DLT input formats:

    % For spatial conv: {SSSC, SSSCB, SSSCT, SSSCBT}. TF input formats are
    % always BSSSC or BTSSSC in these cases, and we export to
    % TimeDistributed conv3d.

    % For spatio-temporal conv: {SSCT, SSCBT}. TF input formats are always
    % BTSSC in these cases, and we export to non-TimeDistributed conv3d.
    % But: In DLT the dimensions to conv over are SST, while in TF they are
    % TSS. So we need to move the strides/padding for T to the front, and
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
            isSpatioTemporal = ismember(this.InputFormat, ["SSCT", "SSCBT"]);
            if isSpatioTemporal
                % Convert hypers from SST to TSS ordering
                Stride         = Stride([3 1 2]);
                FilterSize     = FilterSize([3 1 2]);
                DilationFactor = DilationFactor([3 1 2]);
                PaddingSize    = PaddingSize(:,[3 1 2]);                    % PaddingSize is a 2-by-3 matrix [t l f;b r k]
            end
            % Assemble TF hyperparams
            pool_size = this.Layer.NumFilters;
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
                paddingStr = '';
            end
            if all(DilationFactor==1)
                dilationStr = '';
            else
                dilationStr = sprintf(", dilation_rate=(%s)", join(string(DilationFactor), ','));
            end
            if ~all(PaddingSize(:) == 0) && ~isequal(this.Layer.PaddingValue, 0)
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForConvolutionLayers", this.Layer.Name, string(this.Layer.PaddingValue));
                warningNoBacktrace(this, msg);
                convertedLayer.WarningMessages(end+1) = msg;
            end

            % Generate code
            useTimeDistributed = this.layerAnalyzer.IsTemporal && ~isSpatioTemporal;
            if isequal(this.Layer.PaddingMode, 'same') || all(PaddingSize(:) == 0)
                % Generate only a conv line
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.Conv3D",...
                    "%d, %s%s%s%s", {pool_size, kernel_size, strideStr, paddingStr, dilationStr},...
                    layerName, useTimeDistributed);
            else
                % Generate a padding line and a conv line. 3D padding
                % layout is now a 2-by-3 matrix [t l f;b r k]
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

                    kerasCodeLine(this, padOutputTensorName, this.OutputTensorName, "layers.Conv3D",...
                    "%d, %s%s%s%s", {pool_size, kernel_size, strideStr, paddingStr, dilationStr},...
                    layerName, useTimeDistributed)
                    ];
            end

            % Format of kernel weights:
            % Keras and MATLAB are both: Height-Width-Depth-NumChannels-NumFilters: H x W x D x C x F
            % But we need to permute if isSpatioTemporal
            Weights = this.Layer.Weights;
            if isSpatioTemporal
                % Weights are HWDCF. But D is really T. In TF, T needs to
                % become the first of the spatial dims. So we need to
                % permute to DHWCF. That's [3 1 2 4 5]
                Weights = permute(Weights, [3 1 2 4 5]);
            end
            kerasWeights = permute(Weights, [5 4 3 2 1]);   % Switch memory ordering from Col-major to row-major.
            convertedLayer.LayerName = layerName;
            convertedLayer.weightNames    = ["kernel", "bias"];
            convertedLayer.weightArrays    = {kerasWeights; this.Layer.Bias};                                                 % A matrix and a vector.
            convertedLayer.weightShapes    = {size(Weights, 1:5); numel(this.Layer.Bias)};  	% 5D and 1D.
        end
    end
end
