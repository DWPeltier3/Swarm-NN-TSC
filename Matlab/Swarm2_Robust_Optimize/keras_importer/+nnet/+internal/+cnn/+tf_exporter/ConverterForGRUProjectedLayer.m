classdef ConverterForGRUProjectedLayer < nnet.internal.cnn.tf_exporter.ConverterForGRULayer
    %CONVERTERFORGRUPROJECTEDLAYER Class to export a gruProjectedLayer to
    %Tensorflow

    %   Copyright 2023 The MathWorks, Inc.

    % The "Projected" aspect of this layer is not exported. Instead, we get
    % the full unprojected weights and pass them to the toTensorFlow
    % method of the base class.
    methods
        function convertedLayer = toTensorflow(this)
            fullInputWeights = this.Layer.InputWeights * this.Layer.InputProjector';
            fullRecurrentWeights = this.Layer.RecurrentWeights * this.Layer.OutputProjector';
            % Reuse the inherited method convertGRUWithWeightsToTensorflow.
            convertedLayer = this.convertGRUWithWeightsToTensorflow( ...
                fullInputWeights, fullRecurrentWeights);
            msg = message('nnet_cnn_kerasimporter:keras_importer:exporterConverterForGRUProjectedLayer', this.Layer.Name);
            this.warningNoBacktrace(msg);
            convertedLayer.WarningMessages = msg;
        end
    end
end