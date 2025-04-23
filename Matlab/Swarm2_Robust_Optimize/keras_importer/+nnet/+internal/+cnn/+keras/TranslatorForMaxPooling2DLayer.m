classdef TranslatorForMaxPooling2DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017-2018 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerConfig =
            %   struct with fields:
            %
            %       trainable: 1
            %            name: 'max_pooling2d_1'
            %     data_format: 'channels_last'
            %         padding: 'valid'
            %       pool_size: [2x1 double]
            %         strides: [2x1 double]
            %
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            PoolSize    = fixKeras2DSizeParameter(this, kerasField(LSpec, 'pool_size'));
            Stride      = fixKerasStridesAllowNone(this, kerasField(LSpec, 'strides'), PoolSize);
            switch kerasField(LSpec, 'padding')
                case 'valid'
                    Padding = [0 0];
                case 'same'
                    Padding = 'same';
                otherwise
                    assert(false);
            end
            NNTLayers = { maxPooling2dLayer(PoolSize, 'Stride', Stride, 'Padding', Padding, 'Name', LayerName) };
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
