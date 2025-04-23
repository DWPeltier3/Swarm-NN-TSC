classdef TranslatorForAveragePooling1DLayer < nnet.internal.cnn.keras.LayerTranslator
    % Copyright 2021 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerConfig =
            %   struct with fields:
            %
            %     data_format: 'channels_last'
            %       trainable: 1
            %         strides: 1
            %            name: 'average_pooling1d_1'
            %       pool_size: 3
            %         padding: 'valid'
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            PoolSize    = kerasField(LSpec, 'pool_size');
            Stride      = kerasField(LSpec, 'strides');

            if isequal(Stride, 'None')
                Stride = PoolSize;
            end

            switch kerasField(LSpec, 'padding')
                case 'valid'
                    Padding = [0 0];
                    PaddingValue = 0;
                case 'same'
                    Padding = 'same';
                    PaddingValue = 'mean';
                otherwise
                    assert(false);
            end
            NNTLayers = { averagePooling1dLayer(PoolSize, 'Stride', Stride, 'Padding', Padding, 'PaddingValue', PaddingValue, 'Name', LayerName) };
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if ~ismember(kerasField(LSpec, 'padding'), {'valid', 'same'})
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedPadding', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
    end
end

function iWarningWithoutBacktrace(msgID, varargin)
nnet.internal.cnn.keras.util.warningWithoutBacktrace(msgID, varargin{:})
end

