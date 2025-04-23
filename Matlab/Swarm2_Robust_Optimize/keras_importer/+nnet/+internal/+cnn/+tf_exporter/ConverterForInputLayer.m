classdef ConverterForInputLayer < nnet.internal.cnn.tf_exporter.LayerConverter
    % Class to convert a InputLayer into TensorFlow

    %   Copyright 2023 The MathWorks, Inc.

    % Supported input formats: all
    methods
        function convertedLayer = toTensorflow(this)

            % Extract input format and size from layer directly
            % NB: OutputSize property excludes the BT dimensions which
            % might be a problem 
            fmt = this.Layer.InputFormat;
            sz = this.Layer.InputSize;

            % Error if output formats has a U dim
            % NB: U dims unsupported because it is unclear what TF formats
            % these should correspond to.
            if contains(fmt, 'U')
                error(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedUnspecifiedFormats', this.Layer.Name));
            end

            % Error if output format does not have a C dim
            if ~contains(fmt, 'C')
                error(message('nnet_cnn_kerasimporter:keras_importer:UnsupportedNoCDim', this.Layer.Name));
            end

            % Remove B dimension and corresponding size because it is
            % implicit in Keras input
            if contains(fmt, 'B')
                batchDim = find(fmt=='B');
                sz(batchDim) = [];
                fmt(batchDim) = [];
            end

            % Convert to DLT format ordering
            % NB: Although the DLT input layer can have format orderings
            % that are non-standard, Keras does not auto-permute the data,
            % so data should be provided in a format Keras expects 
            % NB: Reordering means that the format will ALWAYS be S*CT,
            % S*C, CT, C, or T
            [fmt, permIdx] = deep.internal.format.orderFormat(fmt);
            sz = sz(permIdx);

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;

            % Make sure that the time dimension is the first dimension
            % NB: Keras input layer uses 'None' to indicate that it has
            % variable size
            if contains(fmt, 'T')
                % Ensure channel is always the last element
                % NB: Only required if has T dim as it is guaranteed to the
                % trailing dim in all other cases
                channelDim = find(fmt=='C');
                sz([channelDim, end]) = sz([end, channelDim]);
                fmt([channelDim, end]) = fmt([end, channelDim]);

                % Delete T dim from size vector
                sz(fmt=='T') = [];
                convertedLayer.layerCode = sprintf("%s = keras.Input(shape=(None,%s))", ...
                            this.OutputTensorName, join(string(sz), ','));
            else
                convertedLayer.layerCode = sprintf("%s = keras.Input(shape=(%s))", ...
                        this.OutputTensorName, join(string(sz), ','));
            end
        end
    end
end