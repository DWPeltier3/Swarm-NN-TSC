classdef TranslatorForMaxPooling1DLayer < nnet.internal.cnn.keras.LayerTranslator
    % Copyright 2021 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerConfig =
            %   struct with fields:
            %
            %       trainable: 1
            %            name: 'max_pooling1d_1'
            %     data_format: 'channels_last'
            %         padding: 'valid'
            %       pool_size: 2
            %         strides: 1
            %
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            PoolSize    = kerasField(LSpec, 'pool_size');
            Stride      = kerasField(LSpec, 'strides');

            if isequal(Stride, 'None')
                Stride = PoolSize;
            end

            switch kerasField(LSpec, 'padding')
                case 'valid'
                    Padding = [0 0];
                case 'same'
                    Padding = 'same';
                otherwise
                    assert(false);
            end
            NNTLayers = { maxPooling1dLayer(PoolSize, 'Stride', Stride, 'Padding', Padding, 'Name', LayerName) };
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            tf = ismember(kerasField(LSpec, 'padding'), {'valid', 'same'});
            if tf
                Message = '';
            else
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedPadding', LSpec.Name);
            end
        end
    end
end

