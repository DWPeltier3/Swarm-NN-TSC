classdef ConverterForImageInputLayers < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % A converter for imageInputLayer AND image3dImputLayer
    methods
        function convertedLayer = toTensorflow(this)
            imgSize = this.OutputSize{1};

            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch char(this.Layer.Normalization)
                case 'none'
                    convertedLayer.layerCode = sprintf("%s = keras.Input(shape=(%s))", ...
                        this.OutputTensorName, join(string(imgSize), ','));
                case 'rescale-symmetric'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the min and max images as weights. Min and Max
                    % have size [h w c], [1 1 c], or [1 1]. Expand them to
                    % [h w c] in all cases.
                    mlMin           = this.Layer.Min + zeros(imgSize);
                    mlMax           = this.Layer.Max + zeros(imgSize);
                    numDimsImg     = numel(imgSize);
                    kerasMin       = permute(mlMin, numDimsImg:-1:1);           % Convert HW*C from colmaj to rowmaj
                    kerasMax       = permute(mlMax, numDimsImg:-1:1);           % Convert HW*C from colmaj to rowmaj
                    % Rearranging the rescaling formula from:
                    % Scale*(input) + Offset, to:
                    % (input - mean) / variance   
                    kerasMean     = (kerasMax + kerasMin)./2;
                    kerasStd      = (kerasMax - kerasMin)./2;
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["mean", "variance"];
                    convertedLayer.weightArrays     = {kerasMean, kerasStd.*kerasStd};
                    convertedLayer.weightShapes     = {imgSize, imgSize};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(%s), name=""%s"")", intermediateName, join(string(imgSize), ','), intermediateName);
                        sprintf("%s = keras.layers.Normalization(axis=(%s), name=""%s"")(%s)", this.OutputTensorName, join(string(1:numDimsImg), ','), layerName, intermediateName)
                        ];
                case 'rescale-zero-one'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the min and max images as weights. Min and Max
                    % have size [h w c], [1 1 c], or [1 1]. Expand them to
                    % [h w c] in all cases.
                    mlMin           = this.Layer.Min + zeros(imgSize);
                    mlMax           = this.Layer.Max + zeros(imgSize);
                    numDimsImg     = numel(imgSize);
                    kerasMin       = permute(mlMin, numDimsImg:-1:1);           % Convert HW*C from colmaj to rowmaj
                    kerasMax       = permute(mlMax, numDimsImg:-1:1);           % Convert HW*C from colmaj to rowmaj
                    % Rearranging the rescaling formula from:
                    % Scale*(input) + Offset, to:
                    % (input - mean) / variance
                    kerasMean     = kerasMin;
                    kerasStd      = (kerasMax - kerasMin);
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["mean", "variance"];
                    convertedLayer.weightArrays     = {kerasMean, kerasStd.*kerasStd};
                    convertedLayer.weightShapes     = {imgSize, imgSize};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(%s), name=""%s"")", intermediateName, join(string(imgSize), ','), intermediateName);
                        sprintf("%s = keras.layers.Normalization(axis=(%s), name=""%s"")(%s)", this.OutputTensorName, join(string(1:numDimsImg), ','), layerName, intermediateName)
                        ];
                case 'zerocenter'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the mean image as a weight. Mean has size [h w
                    % c], [1 1 c], or [1 1]. Expand it to [h w c] in all
                    % cases.
                    mlMean          = this.Layer.Mean + zeros(imgSize);
                    mlStd           = ones(imgSize);
                    numDimsImg      = numel(imgSize);
                    kerasMean       = permute(mlMean, numDimsImg:-1:1);             % Convert HW*C from colmaj to rowmaj
                    kerasStd        = permute(mlStd, numDimsImg:-1:1);              % Convert HW*C from colmaj to rowmaj
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["mean", "variance"];
                    convertedLayer.weightArrays     = {kerasMean, kerasStd.*kerasStd};
                    convertedLayer.weightShapes     = {imgSize, imgSize};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(%s), name=""%s"")", intermediateName, join(string(imgSize), ','), intermediateName);
                        sprintf("%s = keras.layers.Normalization(axis=(%s), name=""%s"")(%s)", this.OutputTensorName, join(string(1:numDimsImg), ','), layerName, intermediateName)
                        ];
                case 'zscore'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the mean and std images as weights. Mean and Std
                    % have size [h w c], [1 1 c], or [1 1]. Expand them to
                    % [h w c] in all cases.
                    mlMean          = this.Layer.Mean + zeros(imgSize);
                    mlStd           = this.Layer.StandardDeviation + zeros(imgSize);
                    numDimsImg      = numel(imgSize);
                    kerasMean       = permute(mlMean, numDimsImg:-1:1);          % Convert HW*C from colmaj to rowmaj
                    kerasStd        = permute(mlStd, numDimsImg:-1:1);           % Convert HW*C from colmaj to rowmaj
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["mean", "variance"];
                    convertedLayer.weightArrays     = {kerasMean, kerasStd.*kerasStd};
                    convertedLayer.weightShapes     = {imgSize, imgSize};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(%s), name=""%s"")", intermediateName, join(string(imgSize), ','), intermediateName);
                        sprintf("%s = keras.layers.Normalization(axis=(%s), name=""%s"")(%s)", this.OutputTensorName, join(string(1:numDimsImg), ','), layerName, intermediateName)
                        ];
                otherwise
                    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForInputLayers", this.Layer.Name, char(this.Layer.Normalization));
                    warningNoBacktrace(this, msg);
                    convertedLayer.WarningMessages(end+1) = msg;
                    convertedLayer.layerCode = sprintf("%s = keras.Input(shape=(%s))", this.OutputTensorName, join(string(imgSize), ','));
            end
        end
    end
end