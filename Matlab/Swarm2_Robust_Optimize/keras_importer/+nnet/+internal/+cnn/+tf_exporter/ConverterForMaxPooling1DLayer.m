classdef ConverterForMaxPooling1DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

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
    %       (batch_size, steps, features)
    % This DOES NOT MATCH Conv1D, which allows batch_size to be N-D. To
    % handle the BTSC cases, where we want to Max over S, we can apply
    % maxPool2D to Max over TS, and make the T size 1.
    methods
        function convertedLayer = toTensorflow(this)
            pool_size   = this.Layer.PoolSize;     	% scalar
            strides     = this.Layer.Stride;     	% scalar
            % Generate code
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch this.InputFormat
                case {"CBT", "SCB", "CT", "SC"}
                    % In Keras these are BTC or BSC
                    if isequal(this.Layer.PaddingMode, 'same')
                        % Generate only a maxPool line with same padding
                        convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.MaxPool1D",...
                            "pool_size=%d, strides=%d, padding=""same""", {pool_size, strides},...
                            "", false);
                    elseif all(this.Layer.PaddingSize == 0)
                        % Generate only a maxPool line with valid padding
                        convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.MaxPool1D",...
                            "pool_size=%d, strides=%d", {pool_size, strides},...
                            "", false);
                    else
                        % Generate a padding line and a conv line.
                        padBefore = this.Layer.PaddingSize(1);
                        padAfter = this.Layer.PaddingSize(2);
                        padOutputTensorName = sprintf("%s_prepadded", this.OutputTensorName);
                        % ZeroPadding1D always pads the second dim. Shapes
                        % are BTC and BSC, which is what we want.
                        paddingLine = kerasCodeLine(this, this.InputTensorName, padOutputTensorName, "layers.ZeroPadding1D",...
                            "padding=((%d,%d))", {padBefore, padAfter},...
                            "", false);
                        convertedLayer.layerCode = [
                            paddingLine
                            kerasCodeLine(this, padOutputTensorName, this.OutputTensorName, "layers.MaxPool1D",...
                            "pool_size=%d, strides=%d", {pool_size, strides},...
                            this.OutputTensorName, false);
                            ];
                    end
                case {"SCBT", "SCT"}
                    % In Keras these are BTSC.
                    % For the maxPool2D, we make the first element (T) of
                    % the poolsize and strides 1.
                    if isequal(this.Layer.PaddingMode, 'same')
                        % Generate only a maxPool line with same padding
                        convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.MaxPool2D",...
                            "pool_size=(1,%d), strides=(1,%d), padding=""same""", {pool_size, strides},...
                            "", false);
                    elseif all(this.Layer.PaddingSize == 0)
                        % Generate only a maxPool line with valid padding
                        convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.MaxPool2D",...
                            "pool_size=(1,%d), strides=(1,%d)", {pool_size, strides},...
                            "", false);
                    else
                        % Generate a padding line and a conv line.
                        padBefore = this.Layer.PaddingSize(1);
                        padAfter = this.Layer.PaddingSize(2);
                        padOutputTensorName = sprintf("%s_prepadded", this.OutputTensorName);
                        % Here we use 2D padding.
                        % layers.ZeroPadding2D interprets its input
                        % as BHWC. Since the input we're passing is BTSC,
                        % HW here correspond to TS, and we want to pad S.
                        % So top and bottom pads are 0.
                        paddingLine = kerasCodeLine(this, this.InputTensorName, padOutputTensorName, "layers.ZeroPadding2D",...
                            "padding=((0,0),(%d,%d))", {padBefore, padAfter},...
                            "", false);
                        convertedLayer.layerCode = [
                            paddingLine
                            kerasCodeLine(this, padOutputTensorName, this.OutputTensorName, "layers.MaxPool2D",...
                            "pool_size=(1,%d), strides=(1,%d)", {pool_size, strides},...
                            this.OutputTensorName, false);
                            ];
                    end
                otherwise
                    convertedLayer.Success = false;
                    return
            end
        end
    end
end