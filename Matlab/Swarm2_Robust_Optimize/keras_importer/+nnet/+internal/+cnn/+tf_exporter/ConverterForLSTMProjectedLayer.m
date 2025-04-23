classdef ConverterForLSTMProjectedLayer < nnet.internal.cnn.tf_exporter.ConverterForLSTMLayer_Base

    %   Copyright 2022 The MathWorks, Inc.

    % The "Projected" aspect of this layer is not exported. Instead, we get
    % the full unprojected weights and pass them to the toTensorFlow
    % method of the base class.
    methods
        function convertedLayer = toTensorflow(this)
            fullInputWeights = this.Layer.InputWeights * this.Layer.InputProjector';
            fullRecurrentWeights = this.Layer.RecurrentWeights * this.Layer.OutputProjector';
            convertedLayer = toTensorflow@nnet.internal.cnn.tf_exporter.ConverterForLSTMLayer_Base(this, ...
                fullInputWeights, fullRecurrentWeights);
            msg = message('nnet_cnn_kerasimporter:keras_importer:exporterConverterForLSTMProjectedLayer', this.Layer.Name);
            this.warningNoBacktrace(msg);
            convertedLayer.WarningMessages = msg;
        end
    end
end
