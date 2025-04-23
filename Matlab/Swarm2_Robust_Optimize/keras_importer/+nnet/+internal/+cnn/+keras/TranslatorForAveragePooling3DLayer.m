classdef TranslatorForAveragePooling3DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017-2020 The MathWorks, Inc.

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
            %         strides: [3x1 double]
            %            name: 'average_pooling3d_1'
            %       pool_size: [3x1 double]
            %         padding: 'valid'
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            PoolSize    = fixKeras3DSizeParameter(this, kerasField(LSpec, 'pool_size'));
            Stride      = fixKeras3DStridesAllowNone(this, kerasField(LSpec, 'strides'), PoolSize);
            switch kerasField(LSpec, 'padding')
                case 'valid'
                    Padding = [0 0 0;0 0 0];
                    PaddingValue = 0;
                case 'same'
                    Padding = 'same';
                    PaddingValue = 'mean';
                otherwise
                    assert(false);
            end
            NNTLayers = { averagePooling3dLayer(PoolSize, 'Stride', Stride, 'Padding', Padding, 'PaddingValue', PaddingValue, 'Name', LayerName) };
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
