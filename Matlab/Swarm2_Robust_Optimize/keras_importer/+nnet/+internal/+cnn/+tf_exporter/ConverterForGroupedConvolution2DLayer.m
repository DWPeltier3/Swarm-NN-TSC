classdef ConverterForGroupedConvolution2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSC[B][T]
    methods
        function convertedLayer = toTensorflow(this)
            if this.Layer.NumChannelsPerGroup == 1
                convertedLayer = toTensorflow_Depthwise(this);
            else
                convertedLayer = toTensorflow_NotDepthwise(this);
            end
        end
    end

    methods(Access=private)
        function convertedLayer = toTensorflow_Depthwise(this)
            % This applies when this.Layer.NumChannelsPerGroup==1
            assert(this.Layer.NumChannelsPerGroup == 1);

            % It's DepthwiseConvolution
            kernel_sizeStr = sprintf("(%s)", join(string(this.Layer.FilterSize), ','));        % (H,W)
            if all(this.Layer.Stride==1)
                strideStr = "";
            else
                strideStr = sprintf("strides=(%s), ", join(string(this.Layer.Stride), ','));            % (H,W)
            end
            if this.Layer.NumFiltersPerGroup==1
                depth_multiplier = "";
            else
                depth_multiplier = sprintf("depth_multiplier=%d, ", this.Layer.NumFiltersPerGroup);
            end
            if all(this.Layer.DilationFactor==1)
                dilationStr = "";
            else
                dilationStr = sprintf("dilation_rate=(%s), ", join(string(this.Layer.DilationFactor), ','));	% (H,W)
            end
            % Generate code
            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if isequal(this.Layer.PaddingMode, 'same')
                % Generate only a DepthwiseConv2D line with same padding
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.DepthwiseConv2D", ...
                    "kernel_size=%s, padding=""same"", %s%s%suse_bias=True", {kernel_sizeStr, strideStr, depth_multiplier, dilationStr},...
                    layerName, this.layerAnalyzer.IsTemporal);
            elseif isequal(this.Layer.PaddingMode, 'valid') || all(this.Layer.PaddingSize == 0)
                % Generate only a DepthwiseConv2D line
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.DepthwiseConv2D", ...
                    "kernel_size=%s, %s%s%suse_bias=True", {kernel_sizeStr, strideStr, depth_multiplier, dilationStr},...
                    layerName, this.layerAnalyzer.IsTemporal);
            else
                if ~isequal(this.Layer.PaddingValue, 0)
                    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForConvolutionLayers", this.Layer.Name, string(this.Layer.PaddingValue));
                    warningNoBacktrace(this, msg);
                    convertedLayer.WarningMessages(end+1) = msg;
                end
                % Generate a padding line and a DepthwiseConv2D line. This
                % handles explicitly specified padding
                t = this.Layer.PaddingSize(1);
                b = this.Layer.PaddingSize(2);
                l = this.Layer.PaddingSize(3);
                r = this.Layer.PaddingSize(4);
                padOutputTensorName = sprintf("%s_prepadded", this.OutputTensorName);
                convertedLayer.layerCode = [
                    kerasCodeLine(this, this.InputTensorName, padOutputTensorName, "layers.ZeroPadding2D", ...
                    "padding=((%d,%d),(%d,%d))", {t, b, l, r}, "", this.layerAnalyzer.IsTemporal)

                    kerasCodeLine(this, padOutputTensorName, this.OutputTensorName, "layers.DepthwiseConv2D",...
                    "kernel_size=%s, %s%s%suse_bias=True", {kernel_sizeStr, strideStr, depth_multiplier, dilationStr},...
                    layerName, this.layerAnalyzer.IsTemporal)
                    ];
            end
            % Format of kernel weights:
            % MATLAB: Height-Width-NumChannelsPerGroup-NumFiltersPerGroup-NumGroups: H x W x C x F x G
            % But we only support C==1. So it's [HW1FG]
            % keras format is [HWGF]
            % DLT bias shape is [1 1 F G].
            % Keras bias is G*F (1-D).
            kerasWeights = permute(this.Layer.Weights, [1 2 5 4 3]);  % HW1FG --> HWGF1
            kerasWeightShape = size(kerasWeights,1:4);
            kerasWeights = permute(kerasWeights, [4 3 2 1]);                                % Switch memory ordering from Col-major to row-major.
            kerasBias = permute(this.Layer.Bias, [4 3 2 1]);          % [1 1 F G] --> [G F 1 1]
            kerasBias = permute(kerasBias, [2 1]);                                          % [G F] Col-major to row-major ordering.
            kerasBias = kerasBias(:);                                                       % [G F] row-major to G*F.
            convertedLayer.LayerName = layerName;
            convertedLayer.weightNames     = ["kernel", "bias"];
            convertedLayer.weightArrays    = {kerasWeights; kerasBias};
            convertedLayer.weightShapes    = {kerasWeightShape; numel(kerasBias)};  	        % 4D and 1D.
        end

        function convertedLayer = toTensorflow_NotDepthwise(this)
            % This applies when this.Layer.NumChannelsPerGroup ~= 1
            assert(this.Layer.NumChannelsPerGroup ~= 1);

            filters         = this.Layer.NumFiltersPerGroup * this.Layer.NumGroups;
            kernel_size     = sprintf("(%s)", join(string(this.Layer.FilterSize), ','));        % (H,W)
            if all(this.Layer.Stride==1)
                strideStr = "";
            else
                strideStr = sprintf("strides=(%s), ", join(string(this.Layer.Stride), ','));            % (H,W)
            end
            if all(this.Layer.DilationFactor==1)
                dilationStr = "";
            else
                dilationStr = sprintf("dilation_rate=(%s), ", join(string(this.Layer.DilationFactor), ','));	% (H,W)
            end
            groups          = this.Layer.NumGroups;
            % Generate code
            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if isequal(this.Layer.PaddingMode, 'same')
                % Generate only a Conv2D line with same padding
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.Conv2D",...
                    "%d, %s, padding=""same"", %s%sgroups=%d", {filters, kernel_size, strideStr, dilationStr, groups},...
                    layerName, this.layerAnalyzer.IsTemporal);
            elseif isequal(this.Layer.PaddingMode, 'valid') || all(this.Layer.PaddingSize == 0)
                % Generate only a Conv2D line
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.Conv2D",...
                    "%d, %s, %s%sgroups=%d", {filters, kernel_size, strideStr, dilationStr, groups},...
                    layerName, this.layerAnalyzer.IsTemporal);
            else
                if ~isequal(this.Layer.PaddingValue, 0)
                    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForConvolutionLayers", this.Layer.Name, string(this.Layer.PaddingValue));
                    warningNoBacktrace(this, msg);
                    convertedLayer.WarningMessages(end+1) = msg;
                end
                % Generate a padding line and a DepthwiseConv2D line. This
                % handles "same" padding and explicitly specified padding,
                % because in either case, this.Layer.PaddingSize is set.
                t = this.Layer.PaddingSize(1);
                b = this.Layer.PaddingSize(2);
                l = this.Layer.PaddingSize(3);
                r = this.Layer.PaddingSize(4);
                padOutputTensorName = sprintf("%s_prepadded", this.OutputTensorName);
                convertedLayer.layerCode = [
                    kerasCodeLine(this, this.InputTensorName, padOutputTensorName, "layers.ZeroPadding2D",...
                    "padding=((%d,%d),(%d,%d))", {t, b, l, r},...
                    "", this.layerAnalyzer.IsTemporal)

                    kerasCodeLine(this, padOutputTensorName, this.OutputTensorName, "layers.Conv2D",...
                    "%d, %s, %s%sgroups=%d", {filters, kernel_size, strideStr, dilationStr, groups},...
                    layerName, this.layerAnalyzer.IsTemporal)
                    ];
            end
            % Format of kernel weights:
            % DLT: Height-Width-NumChannelsPerGroup-NumFiltersPerGroup-NumGroups: [H, W, C/G, F/G, G]
            % Keras: [H, W, C/G, F]
            [H,W,c,f,G] = size(this.Layer.Weights, 1:5);
            kerasWeights = reshape(this.Layer.Weights, [H,W,c,f*G]);  % [H, W, C/G, F/G, G] --> [H, W, C/G, F]
            kerasWeightShape = size(kerasWeights,1:4);
            kerasWeights = permute(kerasWeights, [4 3 2 1]);                                % Switch memory ordering from Col-major to row-major.
            % DLT bias shape is [1 1 F/G G].
            % Keras bias is F (1-D).
            kerasBias = this.Layer.Bias(:);                           % [1 1 F/G G] --> [F 1]
            convertedLayer.LayerName = layerName;
            convertedLayer.weightNames     = ["kernel", "bias"];
            convertedLayer.weightArrays    = {kerasWeights; kerasBias};
            convertedLayer.weightShapes    = {kerasWeightShape; numel(kerasBias)};  	        % 4D and 1D.
        end
    end
end
