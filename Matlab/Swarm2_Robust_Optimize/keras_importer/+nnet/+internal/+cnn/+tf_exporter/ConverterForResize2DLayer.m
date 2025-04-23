classdef ConverterForResize2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSC[B]

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            % Check NearestRoundingMode
            NearestRoundingMode = this.Layer.NearestRoundingMode;
            switch NearestRoundingMode
                case "round"
                otherwise
                    % Warn and substitute
                    NearestRoundingMode = "round";
                    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterResizeSubst", this.Layer.Name, "NearestRoundingMode", this.Layer.NearestRoundingMode, NearestRoundingMode);
                    warningNoBacktrace(this, msg);
                    convertedLayer.WarningMessages(end+1) = msg;
            end

            % Check GeometricTransformMode
            GeometricTransformMode = this.Layer.GeometricTransformMode;
            switch GeometricTransformMode
                case "half-pixel"
                otherwise
                    % Warn and substitute
                    GeometricTransformMode = "half-pixel";
                    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterResizeSubst", this.Layer.Name, "GeometricTransformMode", this.Layer.GeometricTransformMode, GeometricTransformMode);
                    warningNoBacktrace(this, msg);
                    convertedLayer.WarningMessages(end+1) = msg;
            end

            % Set method
            Method = this.Layer.Method;
            switch Method
                case "nearest"
                    method = "nearest";
                    % Warn about possible 1-pixel error for 'nearest'
                    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterResizeNearest", this.Layer.Name);
                    warningNoBacktrace(this, msg);
                    convertedLayer.WarningMessages(end+1) = msg;
                otherwise
                    % only other possible value is "bilinear"
                    method = "bilinear";
            end

            % Warn if Scale passed
            if ~isempty(this.Layer.Scale)
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

            % Use the DLT layer's OutputSize for TF
            layerOutputSize = this.OutputSize{1}(1:2);            % [H W]

            % If EnableReferenceInput=true, include a layer to bind the
            % second input into the keras model
            if this.Layer.EnableReferenceInput
                tempName = this.OutputTensorName + "_passthrough";
                convertedLayer.layerCode = [
                    sprintf("%s = ignoreInput2Layer()(%s, %s)", tempName, this.InputTensorName(1), this.InputTensorName(2));

                    kerasCodeLine(this, tempName, this.OutputTensorName,...
                    "layers.Resizing", "%d, %d, interpolation='%s'", ...
                    {layerOutputSize(1), layerOutputSize(2), string(method)},...
                    "", this.layerAnalyzer.IsTemporal);
                    ];
                convertedLayer.customLayerNames = "ignoreInput2Layer";
            else
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName(1), this.OutputTensorName,...
                    "layers.Resizing", "%d, %d, interpolation='%s'", ...
                    {layerOutputSize(1), layerOutputSize(2), string(method)},...
                    "", this.layerAnalyzer.IsTemporal);
            end
        end
    end
end