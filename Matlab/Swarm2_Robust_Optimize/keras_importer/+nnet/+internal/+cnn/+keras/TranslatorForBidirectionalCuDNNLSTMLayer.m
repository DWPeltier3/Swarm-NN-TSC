classdef TranslatorForBidirectionalCuDNNLSTMLayer < nnet.internal.cnn.keras.LayerTranslator
    
% Copyright 2019 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {'kernel', 'bias', 'recurrent_kernel'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec
            %   LayerSpec with properties:
            %            Name: 'bidirectional_1'
            %            Type: 'Bidirectional'
            %     KerasConfig: [1◊1 struct]
            %      Translator: []
            %         InConns: {}
            %         Weights: []
            
            % LSpec.KerasConfig
            %   merge_mode: 'concat'
            %        layer: [1◊1 struct]
            %    trainable: 1
            %         name: 'bidirectional_1'
            
            % LSpec.KerasConfig.layer
            %   struct with fields:
            %     class_name: 'CuDNNLSTM'
            %         config: [1◊1 struct]
            
            % LSpec.KerasConfig.layer.config
            %   struct with fields:
            %     recurrent_regularizer: []
            %        kernel_initializer: [1◊1 struct]
            %                      name: 'cu_dnnlstm_3'
            %         kernel_constraint: []
            %          bias_regularizer: []
            %              return_state: 0
            %              go_backwards: 0
            %      activity_regularizer: []
            %                 trainable: 1
            %                  stateful: 0
            %           bias_constraint: []
            %        kernel_regularizer: []
            %     recurrent_initializer: [1◊1 struct]
            %          bias_initializer: [1◊1 struct]
            %                     units: 30
            %          return_sequences: 1
            %          unit_forget_bias: 1
            %      recurrent_constraint: []
            
            LSTMConfig              = LSpec.KerasConfig.layer.config;
            LayerName               = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumHidden               = LSTMConfig.units;
            ReturnSequence          = logical(LSTMConfig.return_sequences);
            Stateful                = true;
            UseBias                 = true;
            Activation              = 'tanh';
            RecurrentActivation     = 'sigmoid';
            
            % Create the CuDNNLSTM using an internal layer:
            name                = LayerName;
            inputSize           = [];
            hiddenSize          = NumHidden;
            rememberCellState   = Stateful;
            rememberHiddenState = Stateful;
            returnSequence      = ReturnSequence;
            activation          = Activation;
            recurrentActivation = RecurrentActivation;
            internalLayer       = nnet.internal.cnn.layer.BiLSTM(name, inputSize, hiddenSize, ...
                rememberCellState, rememberHiddenState, returnSequence, activation, recurrentActivation);
            internalLayer.Bias                              = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            internalLayer.Bias.LearnRateFactor              = single(UseBias);
            if TranslateWeights
                verifyWeights(LSpec, 'forward_cu_dnnlstm_kernel');
                verifyWeights(LSpec, 'backward_cu_dnnlstm_kernel');
                verifyWeights(LSpec, 'forward_cu_dnnlstm_recurrent_kernel');
                verifyWeights(LSpec, 'backward_cu_dnnlstm_recurrent_kernel');
                internalLayer.InputWeights                	= nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
                internalLayer.InputWeights.Value            = [...
                    single(reshapeInputWeight(LSpec.Weights.forward_cu_dnnlstm_kernel))
                    single(reshapeInputWeight(LSpec.Weights.backward_cu_dnnlstm_kernel))
                    ];
                internalLayer.InputWeights.LearnRateFactor 	= single(1);
                internalLayer.InputWeights.L2Factor      	= single(1);
                internalLayer.RecurrentWeights              = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
                internalLayer.RecurrentWeights.Value        = [...
                    single(reshapeRecurrentWeight(LSpec.Weights.forward_cu_dnnlstm_recurrent_kernel))
                    single(reshapeRecurrentWeight(LSpec.Weights.backward_cu_dnnlstm_recurrent_kernel))
                    ];
                internalLayer.RecurrentWeights.LearnRateFactor  = single(1);
                internalLayer.RecurrentWeights.L2Factor     = single(1);
                verifyWeights(LSpec, 'forward_cu_dnnlstm_bias');
                verifyWeights(LSpec, 'backward_cu_dnnlstm_bias');
                internalLayer.Bias.Value                = [...
                    single(reshapeBias(LSpec.Weights.forward_cu_dnnlstm_bias))
                    single(reshapeBias(LSpec.Weights.backward_cu_dnnlstm_bias))
                    ];
                internalLayer.Bias.L2Factor                	= single(1);
                internalLayer.CellState                     = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
                internalLayer.CellState.Value               = zeros(2*NumHidden, 1, 'single');
                internalLayer.HiddenState                   = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
                internalLayer.HiddenState.Value             = zeros(2*NumHidden, 1, 'single');
                internalLayer.InitialCellState            	= zeros(2*NumHidden, 1, 'single');
                internalLayer.InitialHiddenState            = zeros(2*NumHidden, 1, 'single');
            end
            NNTLayers = { nnet.cnn.layer.BiLSTMLayer(internalLayer) };
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            % Check unsupported options:
            if ~isempty(LSpec.KerasConfig.layer.config.activity_regularizer)
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoActivityRegularization', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
end
end

%% Reshape weights
% From the source code on this page:
% https://github.com/keras-team/keras/blob/master/keras/engine/saving.py#L1030
% The weighting scheme of CuDNNLSTM on Keras is different from regular LSTM
% weighting scheme, which needs tranpose and reshape so that it can be fit
% in our implementation.

% merge input and recurrent biases into a single set
function Bm = reshapeBias(Bk)
    half = numel(Bk) / 2;
    Bm = Bk(1:half) + Bk(half+1:end);
end

%  transpose (and reshape) input and recurrent kernels
function Wm = reshapeInputWeight(Wk)
    h = size(Wk,1)/4;
    i = size(Wk,2);
    Wm = [reshape(Wk(1:h,:), [i, h]), reshape(Wk(h+1:2*h,:), [i, h]),...
        reshape(Wk(2*h+1:3*h,:), [i, h]), reshape(Wk(3*h+1:end,:), [i, h]),]';
end

function Wm = reshapeRecurrentWeight(Wk)
    h = size(Wk,1)/4;
    assert(h==floor(h));
    Wm = [Wk(1:h,:), Wk(h+1:2*h,:), Wk(2*h+1:3*h,:), Wk(3*h+1:end,:)]';
end