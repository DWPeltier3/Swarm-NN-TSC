classdef TranslatorForTimeDistributedFlattenLayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2019 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %     trainable: 1
            %          name: 'flatten_1'
            
            % KERAS 2.1.6:
            % LSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %     trainable: 1
            %          name: 'flatten_1'
            %          data_format: 'channels_first'
%             internalLayer = nnet.internal.cnn.layer.Flatten(nnet.internal.cnn.keras.makeNNTName(LSpec.Name), 3);
            NNTLayers = { nnet.keras.layer.TimeDistributedFlattenCStyleLayer(nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
%             NNTLayers = {nnet.cnn.layer.FlattenLayer(internalLayer)};
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if isfield(LSpec.KerasConfig, 'data_format') && isequal(LSpec.KerasConfig.data_format, 'channels_first')
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoChannelsFirst', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
    end
end