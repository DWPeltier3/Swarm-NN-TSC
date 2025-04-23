classdef ConverterForSequenceInputLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)

            if this.Layer.SplitComplexInputs % SplitComplexInputs is not supported
                error(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedSplitComplexInputs', this.Layer.Name));
            end

            inSize = this.OutputSize{1};
            % 'inSize' excludes the BT dimensions. It's S*C (one of C, SC,
            % SSC, SSSC). Keras' Input layer excludes the B dimension, but
            % the T dimension appears as 'None', indicating that it has
            % variable size.
            if numel(inSize)==1
                weightShapeStr = sprintf("(%d,)", inSize);
                mlSize = [inSize 1];
            else
                weightShapeStr = sprintf("(%s)", join(string(inSize),","));
                mlSize = inSize;
            end                
            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            switch char(this.Layer.Normalization)
                case 'none'
                    convertedLayer.layerCode = sprintf("%s = keras.Input(shape=(None,%s))", ...
                        this.OutputTensorName, join(string(inSize), ','));
                case 'rescale-symmetric'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the min and max images as weights.
                    mlMin           = this.Layer.Min + zeros(mlSize);
                    mlMax           = this.Layer.Max + zeros(mlSize);
                    numDimsImg     = numel(mlSize);
                    kerasMin       = permute(mlMin, numDimsImg:-1:1);           % Convert S*C from colmaj to rowmaj
                    kerasMax       = permute(mlMax, numDimsImg:-1:1);           % Convert S*C from colmaj to rowmaj
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["min", "max"];
                    convertedLayer.weightArrays     = {kerasMin, kerasMax};
                    convertedLayer.weightShapes     = {inSize, inSize};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(None,%s))", intermediateName, join(string(inSize), ','))
                        sprintf("%s = RescaleSymmetricLayer(%s, name=""%s"")(%s)", ...
                        this.OutputTensorName, weightShapeStr, layerName, intermediateName)
                        ];
                    % Include a custom layer
                    convertedLayer.customLayerNames = "RescaleSymmetricLayer";
                case 'rescale-zero-one'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the min and max images as weights.
                    mlMin           = this.Layer.Min + zeros(mlSize);
                    mlMax           = this.Layer.Max + zeros(mlSize);
                    numDimsImg     = numel(mlSize);
                    kerasMin       = permute(mlMin, numDimsImg:-1:1);           % Convert S*C from colmaj to rowmaj
                    kerasMax       = permute(mlMax, numDimsImg:-1:1);           % Convert S*C from colmaj to rowmaj
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["min", "max"];
                    convertedLayer.weightArrays     = {kerasMin, kerasMax};
                    convertedLayer.weightShapes     = {inSize, inSize};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(None,%s))", intermediateName, join(string(inSize), ','))
                        sprintf("%s = RescaleZeroOneLayer(%s, name=""%s"")(%s)", ...
                        this.OutputTensorName, weightShapeStr, layerName, intermediateName)
                        ];
                    % Include a custom layer
                    convertedLayer.customLayerNames = "RescaleZeroOneLayer";
                case 'zerocenter'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the mean image as a weight.
                    mlMean          = this.Layer.Mean + zeros(mlSize);
                    numDimsImg      = numel(mlSize);
                    kerasMean       = permute(mlMean, numDimsImg:-1:1);             % Convert S*C from colmaj to rowmaj
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["mean"];
                    convertedLayer.weightArrays     = {kerasMean};
                    convertedLayer.weightShapes     = {inSize};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(None,%s))", intermediateName, join(string(inSize), ','))
                        sprintf("%s = SubtractConstantLayer(%s, name=""%s"")(%s)", ...
                        this.OutputTensorName, weightShapeStr, layerName, intermediateName)
                        ];
                    % Include a custom layer
                    convertedLayer.customLayerNames = "SubtractConstantLayer";
                case 'zscore'
                    intermediateName = sprintf("%s_unnormalized", this.OutputTensorName);
                    convertedLayer.RenameNetworkInputTensor = [this.OutputTensorName, intermediateName];
                    % Save the mean and std images as weights.
                    mlMean          = this.Layer.Mean + zeros(mlSize);
                    mlStd           = this.Layer.StandardDeviation + zeros(mlSize);
                    numDimsImg      = numel(mlSize);
                    kerasMean       = permute(mlMean, numDimsImg:-1:1);             % Convert S*C from colmaj to rowmaj
                    kerasStd        = permute(mlStd, numDimsImg:-1:1);              % Convert S*C from colmaj to rowmaj
                    convertedLayer.LayerName	= layerName;
                    convertedLayer.weightNames     = ["mean", "stdev"];
                    convertedLayer.weightArrays     = {kerasMean, kerasStd};
                    convertedLayer.weightShapes     = {inSize, inSize};
                    % Generate code
                    convertedLayer.layerCode = [...
                        sprintf("%s = keras.Input(shape=(None,%s))", intermediateName, join(string(inSize), ','))
                        sprintf("%s = ZScoreLayer(%s, name=""%s"")(%s)", ...
                        this.OutputTensorName, weightShapeStr, layerName, intermediateName)
                        ];
                    % Include a custom layer
                    convertedLayer.customLayerNames = "ZScoreLayer";
                otherwise
                    msg = message("nnet_cnn_kerasimporter:keras_importer:exporterConverterForInputLayers", this.Layer.Name, char(this.Layer.Normalization));
                    warningNoBacktrace(this, msg);
                    convertedLayer.WarningMessages(end+1) = msg;
                    convertedLayer.layerCode = sprintf("%s = keras.Input(shape=(None,%s))", ...
                        this.OutputTensorName, join(string(inSize), ','));
            end
        end
    end
end
