classdef ConverterForResize3DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSSC[B]

    % There is no 3D resizing layer or function in TensorFlow. The only
    % case we support is integer upscaling with the 'nearest' method, in
    % which case we can use the tf.keras.layers.UpSampling3D layer.

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            % Require 'nearest' method, and integer > 1 effective scales
            layerInputSize = this.InputSize{1}(1:3);                % [H W D]
            layerOutputSize = this.OutputSize{1}(1:3);              % [H W D]
            scales = layerOutputSize ./ layerInputSize;
            if this.Layer.Method ~= "nearest" || ...
                    any(scales < 1) || any(floor(scales) ~= scales)
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForResize3DLayer", this.Layer.Name);
                warningNoBacktrace(this, msg);
                convertedLayer.WarningMessages(end+1) = msg;
                convertedLayer.Success = false;
                return;
            end

            % Warn if non-integer Scale passed
            if ~isempty(this.Layer.Scale) && any(this.Layer.Scale ~= floor(this.Layer.Scale))
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterResizeScale", this.Layer.Name);
                warningNoBacktrace(this, msg);
                convertedLayer.WarningMessages(end+1) = msg;
            end

            % Warn if OutputSize passed with a NaN
            if any(isnan(this.Layer.OutputSize))
                msg = message("nnet_cnn_kerasimporter:keras_importer:exporterResizeOutputSizeNaN", this.Layer.Name);
                warningNoBacktrace(this, msg);
                convertedLayer.WarningMessages(end+1) = msg;
            end

            % If EnableReferenceInput=true, include a layer to bind the
            % second input into the keras model
            if this.Layer.EnableReferenceInput
                tempName = this.OutputTensorName + "_passthrough";
                convertedLayer.layerCode = [
                    sprintf("%s = ignoreInput2Layer()(%s, %s)", tempName, this.InputTensorName(1), this.InputTensorName(2));

                    kerasCodeLine(this, tempName, this.OutputTensorName,...
                    "layers.UpSampling3D", "size=(%d,%d,%d)", ...
                    {scales(1), scales(2), scales(3)},...
                    "", this.layerAnalyzer.IsTemporal);
                    ];
                convertedLayer.customLayerNames = "ignoreInput2Layer";
            else
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName(1), this.OutputTensorName,...
                    "layers.UpSampling3D", "size=(%d,%d,%d)", ...
                    {scales(1), scales(2), scales(3)},...
                    "", this.layerAnalyzer.IsTemporal);
            end
        end
    end
end