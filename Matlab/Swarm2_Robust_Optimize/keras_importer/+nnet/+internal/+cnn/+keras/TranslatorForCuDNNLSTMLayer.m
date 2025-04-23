classdef TranslatorForCuDNNLSTMLayer < nnet.internal.cnn.keras.LayerTranslator
    
%   Copyright 2019 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {'kernel', 'bias', 'recurrent_kernel'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec.KerasConfig
            % ans = 
            % 
            %   struct with fields:
            % 
            %     recurrent_regularizer: []
            %        kernel_initializer: [1◊1 struct]
            %                      name: 'cu_dnnlstm_1'
            %         kernel_constraint: []
            %          bias_regularizer: []
            %              return_state: 0
            %                     dtype: 'float32'
            %      activity_regularizer: []
            %                 trainable: 1
            %                  stateful: 0
            %           bias_constraint: []
            %        kernel_regularizer: []
            %          bias_initializer: [1◊1 struct]
            %     recurrent_initializer: [1◊1 struct]
            %              go_backwards: 0
            %                     units: 10
            %          return_sequences: 0
            %         batch_input_shape: [3◊1 double]
            %      recurrent_constraint: []
            %          unit_forget_bias: 1
            
            
            % From the source code on this page:
            % https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L328
            % there's no customized activation function or recurrent activation
            % function, the default activation function would be tanh and the
            % default recurrent activation function would be sigmoid

            LayerName               = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumHidden               = kerasField(LSpec, 'units');
            ReturnSequence          = logical(kerasField(LSpec, 'return_sequences'));
            Stateful                = true;
            UseBias                 = true;
            Activation              = 'tanh';
            RecurrentActivation     = 'sigmoid';

            % Create the CuDNNLSTM using an internal layer
            name                = LayerName;
            inputSize           = [];
            hiddenSize          = NumHidden;
            rememberCellState   = Stateful;
            rememberHiddenState = Stateful;
            returnSequence      = ReturnSequence;
            activation          = Activation;
            recurrentActivation = RecurrentActivation;
            internalLayer       = nnet.internal.cnn.layer.LSTM(name, inputSize, hiddenSize, ...
                rememberCellState, rememberHiddenState, returnSequence, activation, recurrentActivation);
            internalLayer.Bias                              = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            internalLayer.Bias.LearnRateFactor              = single(UseBias);
            internalLayer.Bias.L2Factor                  	= single(0);
            internalLayer.InputWeights.LearnRateFactor      = single(1);
            internalLayer.InputWeights.L2Factor             = single(1);
            if TranslateWeights
                verifyWeights(LSpec, 'kernel');
                verifyWeights(LSpec, 'recurrent_kernel');
                internalLayer.InputWeights                	= nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
                internalLayer.InputWeights.Value            = single(reshapeInputWeight(LSpec.Weights.kernel));
                internalLayer.InputWeights.LearnRateFactor 	= single(1);
                internalLayer.InputWeights.L2Factor      	= single(1);
                internalLayer.RecurrentWeights              = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
                internalLayer.RecurrentWeights.Value        = single(reshapeRecurrentWeight(LSpec.Weights.recurrent_kernel));
                internalLayer.RecurrentWeights.LearnRateFactor  = single(1);
                internalLayer.RecurrentWeights.L2Factor     = single(1);
                verifyWeights(LSpec, 'bias');
                internalLayer.Bias.Value                    = single(reshapeBias(LSpec.Weights.bias));
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

