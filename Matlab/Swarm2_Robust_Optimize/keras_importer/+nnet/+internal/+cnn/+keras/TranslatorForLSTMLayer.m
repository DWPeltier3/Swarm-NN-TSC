classdef TranslatorForLSTMLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017-2023 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {'kernel', 'bias', 'recurrent_kernel'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %            implementation: 0
            %           bias_constraint: []
            %         batch_input_shape: [3×1 double]
            %              go_backwards: 0
            %                   dropout: 0
            %                 trainable: 1
            %      recurrent_constraint: []
            %                     units: 30
            %                activation: 'tanh'
            %     recurrent_regularizer: []
            %      recurrent_activation: 'hard_sigmoid'
            %                     dtype: 'float32'
            %     recurrent_initializer: [1×1 struct]
            %                  use_bias: 1
            %         kernel_constraint: []
            %                      name: 'lstm_1'
            %                  stateful: 0
            %          unit_forget_bias: 1
            %          bias_initializer: [1×1 struct]
            %          bias_regularizer: []
            %        kernel_regularizer: []
            %          return_sequences: 0
            %                    unroll: 0
            %      activity_regularizer: []
            %         recurrent_dropout: 0
            %        kernel_initializer: [1×1 struct]
            
            % From the source code on this page:
            % https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L869,
            % we find in function "step" (approx lines 1122-1168) that
            % 'recurrent_activation' is used for all 3 gates (i,f,o here.
            % "sigma" in the Search Space Odyssey paper).  And 'activation'
            % is used for both the input and output activations ("g" and
            % "h").
            LayerName               = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumHidden               = kerasField(LSpec, 'units');
            ReturnSequence          = logical(kerasField(LSpec, 'return_sequences'));
            Stateful                = true;
            UseBias                 = logical(kerasField(LSpec, 'use_bias'));
            if isempty(UseBias)
                UseBias = false;    % Empty means false for this field.
            end
            Activation              = mapActivation(kerasField(LSpec, 'activation'));
            RecurrentActivation     = mapRecurrentActivation(kerasField(LSpec, 'recurrent_activation'));
            
            % Create the LSTM using an internal layer:
            name = LayerName;
            inputSize           = [];
            hiddenSize          = NumHidden;
            rememberCellState   = Stateful;
            rememberHiddenState = Stateful;
            returnSequence      = ReturnSequence;
            activation          = Activation;
            recurrentActivation = RecurrentActivation;
            internalLayer       = nnet.internal.cnn.layer.LSTM(name, inputSize, hiddenSize, ...
                rememberCellState, rememberHiddenState, returnSequence, activation, recurrentActivation);
            internalLayer.Bias.LearnRateFactor              = single(UseBias);
            internalLayer.Bias.L2Factor                  	= single(0);
            internalLayer.InputWeights.LearnRateFactor 	= single(1);
            internalLayer.InputWeights.L2Factor      	= single(1);
            if TranslateWeights
                verifyWeights(LSpec, 'kernel');
                verifyWeights(LSpec, 'recurrent_kernel');
                internalLayer.InputWeights.Value            = single(reorderLSTMWeights(LSpec.Weights.kernel));
                internalLayer.InputWeights.LearnRateFactor 	= single(1);
                internalLayer.InputWeights.L2Factor      	= single(1);
                internalLayer.RecurrentWeights.Value        = single(reorderLSTMWeights(LSpec.Weights.recurrent_kernel));
                internalLayer.RecurrentWeights.LearnRateFactor  = single(1);
                internalLayer.RecurrentWeights.L2Factor     = single(1);
                if UseBias
                    verifyWeights(LSpec, 'bias');
                    internalLayer.Bias.Value                = single(reorderLSTMWeights(LSpec.Weights.bias));
                else
                    internalLayer.Bias.Value                = zeros(NumHidden, 1, 'single');
                end
               % internalLayer.Bias.L2Factor                	= single(1);
                internalLayer.CellState                     = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
                internalLayer.CellState.Value               = zeros(NumHidden, 1, 'single');
                internalLayer.HiddenState                   = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
                internalLayer.HiddenState.Value             = zeros(NumHidden, 1, 'single');
                internalLayer.InitialCellState            	= zeros(NumHidden, 1, 'single');
                internalLayer.InitialHiddenState            = zeros(NumHidden, 1, 'single');
            end
            NNTLayers = { nnet.cnn.layer.LSTMLayer(internalLayer) };

            
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            % Check unsupported options:
            AName   = mapActivation(kerasField(LSpec, 'activation'));
            RAName  = mapRecurrentActivation(kerasField(LSpec, 'recurrent_activation'));
            if ~isempty(kerasField(LSpec, 'activity_regularizer'))
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoActivityRegularization', LSpec.Name);
            elseif isempty(AName)
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:LSTMActivation', kerasField(LSpec, 'activation'));
            elseif isempty(RAName)
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:LSTMRecurrentActivation', kerasField(LSpec, 'recurrent_activation'));
            elseif logical(kerasField(LSpec, 'go_backwards'))
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
    case 'relu'
        DLTName = 'relu';
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
