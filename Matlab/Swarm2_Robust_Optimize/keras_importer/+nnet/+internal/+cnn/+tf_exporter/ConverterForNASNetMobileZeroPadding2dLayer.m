classdef ConverterForNASNetMobileZeroPadding2dLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = layers.ZeroPadding2D(padding=((%d,%d),(%d,%d)))(%s)", ...
                this.OutputTensorName, this.Layer.Top, this.Layer.Bottom, this.Layer.Left, this.Layer.Right, this.InputTensorName);  
        end
    end
end
