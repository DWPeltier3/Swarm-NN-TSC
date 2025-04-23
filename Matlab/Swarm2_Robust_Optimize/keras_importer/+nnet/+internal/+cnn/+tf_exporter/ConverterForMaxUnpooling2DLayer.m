classdef ConverterForMaxUnpooling2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.Success = false;
            return
            %             error(message("nnet_cnn_kerasimporter:keras_importer:exporterUnsupportedLayer", layerAnalyzer.Name, "MaxUnpooling2DLayer"));


            % Find some properties of the matching maxPooling2dLayer
            portName = this.InputTensorName(1);
            maxPoolLayerName = extractBefore(portName, strlength(portName)-3);   % Remove "_out" from the end.
            [tf, idx] = ismember(maxPoolLayerName, {this.networkAnalysis.Net.Layers.Name});
            assert(tf);
            maxPoolLayer = this.networkAnalysis.Net.Layers(idx);
            pool_size = "(" + join(string(maxPoolLayer.PoolSize), ",") + ")";
            strides   = "(" + join(string(maxPoolLayer.Stride), ",") + ")";
            if isequal(maxPoolLayer.PaddingMode, 'same')
                padding = "SAME";
            else
                padding = "VALID";
            end
            % Generate code
            convertedLayer.layerCode = sprintf("%s = tfa.layers.MaxUnpooling2D(pool_size=%s, strides=%s, padding=""%s"")(%s, %s)", ...
                this.OutputTensorName, pool_size, strides, padding, this.InputTensorName(1), this.InputTensorName(2));
            convertedLayer.packagesNeeded = "tfa";
        end
    end
end