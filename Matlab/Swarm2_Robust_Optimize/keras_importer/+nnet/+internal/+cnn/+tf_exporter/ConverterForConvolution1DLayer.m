classdef ConverterForConvolution1DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % According to the DLT Doc, this layer can handle these input formats:
    % in DLT:                   in Keras:
    % CBT (pool over T)         BTC
    % SCB (pool over S)         BSC
    % SCBT (pool over S)        BTSC
    %
    % ...And these dlnetwork-only formats:
    %
    % CT                        BTC
    % SC                        BSC
    % SCT                       BTSC

    % According to the keras doc, the input is always interpreted as:
    %       batch_shape + (steps, input_dim)
    % That means the LAST TWO dimensions are always interpreted as TC, no
    % matter what they really are, and all preceding dims are considered
    % the batch shape and treated independently. Also, Keras 1D conv and
    % pooling always operate over what they think are the "T" (next to
    % last) dimension. That implies that an ordinary call to
    % keras.Conv1D(BSC) convolves over the S dimension, which is what we
    % want to match DLT. Similarly, keras.Conv1D(BTSC) does the same.
    methods
        function convertedLayer = toTensorflow(this)
            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            numFilters = this.Layer.NumFilters;
            kernel_size = this.Layer.FilterSize;
            if this.Layer.Stride==1
                strideStr = '';
            else
                strideStr = sprintf(", strides=(%d)", this.Layer.Stride);
            end
            if isequal(this.Layer.PaddingMode, 'same')
                paddingStr = ', padding="same"';
            elseif isequal(this.Layer.PaddingMode, 'causal')
                paddingStr = ', padding="causal"';
            else
                % padding="valid". omit
                paddingStr = '';                                                % A padding layer will be added below.
            end
            if this.Layer.DilationFactor==1
                dilationStr = '';
            else
                dilationStr = sprintf(", dilation_rate=%d", this.Layer.DilationFactor);
            end
            if ~all(this.Layer.PaddingSize == 0) && ~isequal(this.Layer.PaddingValue, 0)
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForConvolutionLayers", this.Layer.Name, string(this.Layer.PaddingValue));
                warningNoBacktrace(this, msg);
                convertedLayer.WarningMessages(end+1) = msg;
            end

            % Generate code
            if isequal(this.Layer.PaddingMode, 'same') || isequal(this.Layer.PaddingMode, 'causal') || all(this.Layer.PaddingSize == 0)
                % Generate only a conv line
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.Conv1D",...
                    "%d, %d%s%s%s", {numFilters, kernel_size, strideStr, paddingStr, dilationStr},...
                    layerName, false);
            else
                % Generate a padding line and a conv line.
                padBefore = this.Layer.PaddingSize(1);
                padAfter = this.Layer.PaddingSize(2);
                padOutputTensorName = sprintf("%s_prepadded", this.OutputTensorName);
                % Depending on the input format, we need to generate
                % different padding layer types
                switch this.InputFormat
                    case {"CBT", "SCB"}
                        % ZeroPadding1D always pads the second dim. Shapes
                        % are BTC and BSC, which is what we want.
                        paddingLine = kerasCodeLine(this, this.InputTensorName, padOutputTensorName, "layers.ZeroPadding1D",...
                            "padding=((%d,%d))", {padBefore, padAfter},...
                            "", false);
                    case "SCBT"
                        % Here we use 2D padding.
                        % layers.ZeroPadding2D interprets its input
                        % as BHWC. Since the input we're passing is BTSC,
                        % HW here correspond to TS, and we want to pad S.
                        % So top and bottom pads are 0
                        paddingLine = kerasCodeLine(this, this.InputTensorName, padOutputTensorName, "layers.ZeroPadding2D",...
                            "padding=((0,0),(%d,%d))", {padBefore, padAfter},...
                            "", false);
                    otherwise
                        convertedLayer.Success = false;
                        return
                end
                convertedLayer.layerCode = [
                    paddingLine

                    kerasCodeLine(this, padOutputTensorName, this.OutputTensorName, "layers.Conv1D",...
                    "%d, %d%s%s%s", {numFilters, kernel_size, strideStr, paddingStr, dilationStr},...
                    layerName, false);
                    ];
            end
            % Format of kernel weights:
            % Keras and MATLAB are both: Height-Width-NumChannels-NumFilters: T x C x F
            kerasWeights = permute(this.Layer.Weights, [3 2 1]);   % Switch memory ordering from Col-major to row-major.
            convertedLayer.LayerName = layerName;
            convertedLayer.weightNames    = ["kernel", "bias"];
            convertedLayer.weightArrays    = {kerasWeights; this.Layer.Bias};                                             % A matrix and a vector.
            convertedLayer.weightShapes    = {size(this.Layer.Weights, 1:3); numel(this.Layer.Bias)};  	% 3D and 1D.
        end
    end
end
