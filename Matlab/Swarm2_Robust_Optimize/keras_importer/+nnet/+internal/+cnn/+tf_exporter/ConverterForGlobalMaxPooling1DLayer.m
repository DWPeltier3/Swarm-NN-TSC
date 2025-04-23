classdef ConverterForGlobalMaxPooling1DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

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
    % handle the BTSC cases, where we want to max over S, we can apply
    % maxPool2D to max over TS, and make the T size 1.
    methods
        function convertedLayer = toTensorflow(this)
            % Generate code
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch this.InputFormat
                case {"CBT", "CT"}
                    % In Keras these are BTC. DLT does NOT keepdims in this
                    % case
                    convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.GlobalMaxPool1D",...
                        "keepdims=False", {}, "", false);
                case {"SCB", "SC"}
                    % In Keras these are BSC. DLT DOES keepdims in this
                    % case
                    convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName, "layers.GlobalMaxPool1D",...
                        "keepdims=True", {}, "", false);
                case {"SCBT", "SCT"}
                    % In Keras these are BTSC. To implement globalmaxPool1D
                    % over S using MaxPool2D, we make the poolsize in S
                    % full-size, and the poolsize in T 1.
                    convertedLayer.layerCode = [
                        sprintf("BTSC = %s.shape", this.InputTensorName)
                        sprintf("%s = layers.MaxPool2D(pool_size=(1,BTSC[2]))(%s)", this.OutputTensorName, this.InputTensorName)
                        ];
                otherwise
                    convertedLayer.Success = false;
                    return
            end
        end
    end
end