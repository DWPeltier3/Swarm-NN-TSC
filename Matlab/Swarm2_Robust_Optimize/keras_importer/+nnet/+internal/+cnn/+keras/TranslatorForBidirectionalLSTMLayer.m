classdef TranslatorForBidirectionalLSTMLayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2018-2023 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {'kernel', 'bias', 'recurrent_kernel'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec =
            %   LayerSpec with properties:
            %            Name: 'bidirectional_16'
            %            Type: 'Bidirectional'
            %     KerasConfig: [1×1 struct]
            %      Translator: []
            %         InConns: {}
            %         Weights: []
            
            % LSpec.KerasConfig
            %   struct with fields:
            %                  name: 'bidirectional_16'
            %             trainable: 1
            %     batch_input_shape: [3×1 double]
            %                 dtype: 'float32'
            %                 layer: [1×1 struct]
            %            merge_mode: 'concat'
            
            % LSpec.KerasConfig.layer
            %   struct with fields:
            %     class_name: 'LSTM'
            %         config: [1×1 struct]
            
            % LSpec.KerasConfig.layer.config
            %   struct with fields:
            %                      name: 'lstm_28'
            %                 trainable: 1
            %          return_sequences: 1
            %              return_state: 0
            %              go_backwards: 0
            %                  stateful: 0
            %                    unroll: 0
            %                     units: 13
            %                activation: 'tanh'
            %      recurrent_activation: 'hard_sigmoid'
            %                  use_bias: 1
            %        kernel_initializer: [1×1 struct]
            %     recurrent_initializer: [1×1 struct]
            %          bias_initializer: [1×1 struct]
            %          unit_forget_bias: 1
            %        kernel_regularizer: []
            %     recurrent_regularizer: []
            %          bias_regularizer: []
            %      activity_regularizer: []
            %         kernel_constraint: []
            %      recurrent_constraint: []
            %           bias_constraint: []
            %                   dropout: 0
            %         recurrent_dropout: 0
            %            implementation: 1
            
            LSTMConfig              = LSpec.KerasConfig.layer.config;
            LayerName               = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumHidden               = LSTMConfig.units;
            ReturnSequence          = logical(LSTMConfig.return_sequences);
            Stateful                = true;
            UseBias                 = logical(LSTMConfig.use_bias);
            if isempty(UseBias)
                UseBias = false;    % Empty means false for this field.
            end
            Activation              = mapActivation(LSTMConfig.activation);
            RecurrentActivation     = mapRecurrentActivation(LSTMConfig.recurrent_activation);
            
            % Create the LSTM using an internal layer:
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
            internalLayer.Bias.LearnRateFactor              = single(UseBias);
            if TranslateWeights
                verifyWeights(LSpec, 'forward_lstm_kernel');
                verifyWeights(LSpec, 'backward_lstm_kernel');
                verifyWeights(LSpec, 'forward_lstm_recurrent_kernel');
                verifyWeights(LSpec, 'backward_lstm_recurrent_kernel');
                internalLayer.InputWeights.Value            = [...
                    single(reorderLSTMWeights(LSpec.Weights.forward_lstm_kernel))
                    single(reorderLSTMWeights(LSpec.Weights.backward_lstm_kernel))
                    ];
                internalLayer.InputWeights.LearnRateFactor 	= single(1);
                internalLayer.InputWeights.L2Factor      	= single(1);
                internalLayer.RecurrentWeights.Value        = [...
                    single(reorderLSTMWeights(LSpec.Weights.forward_lstm_recurrent_kernel))
                    single(reorderLSTMWeights(LSpec.Weights.backward_lstm_recurrent_kernel))
                    ];
                internalLayer.RecurrentWeights.LearnRateFactor  = single(1);
                internalLayer.RecurrentWeights.L2Factor     = single(1);
                if UseBias
                    verifyWeights(LSpec, 'forward_lstm_bias');
                    verifyWeights(LSpec, 'backward_lstm_bias');
                    internalLayer.Bias.Value                = [...
                        single(reorderLSTMWeights(LSpec.Weights.forward_lstm_bias))
                        single(reorderLSTMWeights(LSpec.Weights.backward_lstm_bias))
                        ];
                else
                    internalLayer.Bias.Value                = zeros(8*NumHidden, 1, 'single');
                end
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
            AName   = mapActivation(LSpec.KerasConfig.layer.config.activation);
            RAName  = mapRecurrentActivation(LSpec.KerasConfig.layer.config.recurrent_activation);
            if ~isempty(LSpec.KerasConfig.layer.config.activity_regularizer)
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoActivityRegularization', LSpec.Name);
            elseif isempty(AName)
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:biLSTMActivation', LSpec.KerasConfig.layer.config.activation);
            elseif isempty(RAName)
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:biLSTMRecurrentActivation', LSpec.KerasConfig.layer.config.recurrent_activation);
            elseif logical(LSpec.KerasConfig.layer.config.go_backwards)
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:LSTMGoBackwards');
            else
                tf = true;
                Message = '';
            end
        end
end
end

function DLTName = mapRecurrentActivation(KerasName)
switch KerasName
    case 'sigmoid'
        DLTName = 'sigmoid';
    case 'hard_sigmoid'
        DLTName = 'hard-sigmoid';
    otherwise
        DLTName = '';
end
end

function DLTName = mapActivation(KerasName)
switch KerasName
    case 'tanh'
        DLTName = 'tanh';
    case 'softsign'
        DLTName = 'softsign';
    otherwise
        DLTName = '';
end
end

function Wm = reorderLSTMWeights(Wk)
Wm = Wk;
%% Previous version of NNT had the weights in a different order:
% % LSTM weights in Keras are stored in the order:
% %     [inputGate, forgetGate, Cell,       outputGate],
% % while in NNT the ordering is:
% %     [Cell,      inputGate,  forgetGate, outputGate].
% h = size(Wk,1)/4;
% assert(h==floor(h));
% IdxVec = [2*h+1:3*h, 1:h, h+1:2*h, 3*h+1:4*h]';  % The new block order is [3 1 2 4].
% Wm = Wk(IdxVec,:);
end
