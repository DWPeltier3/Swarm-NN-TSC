classdef TranslatorForEmbeddingLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {'embeddings'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec.KerasConfig
            %   struct with fields:
            %                       name: 'embedding_2'
            %                  trainable: 1
            %          batch_input_shape: [2×1 double]
            %                      dtype: 'float32'
            %                  input_dim: 101
            %                 output_dim: 17
            %     embeddings_initializer: [1×1 struct]
            %     embeddings_regularizer: []
            %       activity_regularizer: []
            %      embeddings_constraint: []
            %                  mask_zero: 0
            %               input_length: 11
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            dimension 	= kerasField(LSpec, 'output_dim');
            numWords    = kerasField(LSpec, 'input_dim');
            % Create layer
            Layer = wordEmbeddingLayer(dimension, numWords, 'Name', LayerName);
            if TranslateWeights
                verifyWeights(LSpec, 'embeddings');
                W = single(LSpec.Weights.embeddings);
%                 % Append zero column on the right
%                 W(:, end+1) = 0;
                Layer.Weights = W;
            end
            NNTLayers = {Layer};
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            % Check unsupported options:
            if ~isempty(kerasField(LSpec, 'activity_regularizer'))
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoActivityRegularization', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
    end
end

