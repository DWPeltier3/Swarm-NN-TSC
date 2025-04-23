classdef ConverterForFeatureInputLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)

            if this.Layer.SplitComplexInputs % SplitComplexInputs is not supported 
                error(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedSplitComplexInputs', this.Layer.Name));
            end
            
            numFeatures = this.OutputSize{1};

            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch char(this.Layer.Normalization)
                case 'none'
                    convertedLayer.layerCode = sprintf("%s = keras.Input(shape=(%d,))", this.OutputTensorName, numFeatures);
                case 'rescale-symmetric'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the min and max images as weights. Min and Max
                    % have size [1 c], or [1 1]. Reshape them to
                    % [c 1] in all cases.
                    TFShape         = numFeatures;
                    mlMin           = this.Layer.Min(:) + zeros([TFShape,1]);      % Note the expansion to MATLAB size here and in subsequent similar lines.
                    mlMax           = this.Layer.Max(:) + zeros([TFShape,1]);
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["min", "max"];
                    convertedLayer.weightArrays     = {mlMin, mlMax};
                    convertedLayer.weightShapes     = {TFShape, TFShape};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(%d,), name=""%s"")", intermediateName, numFeatures, intermediateName);
                        sprintf("%s = RescaleSymmetricLayer((%d,), name=""%s"")(%s)", this.OutputTensorName, numFeatures, layerName, intermediateName)
                        ];
                    % Include a custom layer
                    convertedLayer.customLayerNames = "RescaleSymmetricLayer";
                case 'rescale-zero-one'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the min and max images as weights. Min and Max
                    % have size [1 c], or [1 1]. Reshape them to
                    % [c 1] in all cases.
                    TFShape         = numFeatures;
                    mlMin           = this.Layer.Min(:) + zeros([TFShape,1]);
                    mlMax           = this.Layer.Max(:) + zeros([TFShape,1]);
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["min", "max"];
                    convertedLayer.weightArrays     = {mlMin, mlMax};
                    convertedLayer.weightShapes     = {TFShape, TFShape};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(%d,), name=""%s"")", intermediateName, numFeatures, intermediateName);
                        sprintf("%s = RescaleZeroOneLayer((%d,), name=""%s"")(%s)", this.OutputTensorName, numFeatures, layerName, intermediateName)
                        ];
                    % Include a custom layer
                    convertedLayer.customLayerNames = "RescaleZeroOneLayer";
                case 'zerocenter'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    TFShape         = numFeatures;
                    mlMean          = this.Layer.Mean(:) + zeros([TFShape,1]);
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = "mean";
                    convertedLayer.weightArrays     = {mlMean};
                    convertedLayer.weightShapes     = {TFShape};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(%d,), name=""%s"")", intermediateName, numFeatures, intermediateName);
                        sprintf("%s = SubtractConstantLayer((%d,), name=""%s"")(%s)", this.OutputTensorName, numFeatures, layerName, intermediateName)
                        ];
                    % Include a custom layer
                    convertedLayer.customLayerNames = "SubtractConstantLayer";
                case 'zscore'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    TFShape         = numFeatures;
                    mlMean          = this.Layer.Mean(:) + zeros([TFShape,1]);
                    mlStd           = this.Layer.StandardDeviation(:) + zeros([TFShape,1]);
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["mean", "stdev"];
                    convertedLayer.weightArrays     = {mlMean, mlStd};
                    convertedLayer.weightShapes     = {TFShape, TFShape};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(%d,), name=""%s"")", intermediateName, numFeatures, intermediateName);
                        sprintf("%s = ZScoreLayer((%d,), name=""%s"")(%s)", this.OutputTensorName, numFeatures, layerName, intermediateName)
                        ];
                    % Include a custom layer
                    convertedLayer.customLayerNames = "ZScoreLayer";
                otherwise
                    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForInputLayers", this.Layer.Name, char(this.Layer.Normalization));
                    warningNoBacktrace(this, msg);
                    convertedLayer.WarningMessages(end+1) = msg;
                    convertedLayer.layerCode = sprintf("%s = keras.Input(shape=(%d,))", this.OutputTensorName, numFeatures);
            end
        end
    end
end
