classdef TranslatorForCuDNNGRULayer < nnet.internal.cnn.keras.LayerTranslator
    
%   Copyright 2020 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {'kernel', 'bias', 'recurrent_kernel'};
    end
    
    methods
        function NNTLayers = translate(~, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LSpec.KerasConfig
            % ans = 
            % 
            %   struct with fields:
            % 
            %     recurrent_regularizer: []
            %        kernel_initializer: [1x1 struct]
            %                      name: 'cu_dnngru_1'
            %         kernel_constraint: []
            %          bias_regularizer: []
            %              return_state: 0
            %                     dtype: 'float32'
            %      activity_regularizer: []
            %                 trainable: 1
            %                  stateful: 0
            %           bias_constraint: []
            %        kernel_regularizer: []
            %          bias_initializer: [1x1 struct]
            %     recurrent_initializer: [1x1 struct]
            %              go_backwards: 0
            %                     units: 10
            %          return_sequences: 0
            %         batch_input_shape: [3x1 double]
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
            ResetGateMode           = 'recurrent-bias-after-multiplication';
            
            % Create the CuDNNGRU using an internal layer
            name                = LayerName;
            inputSize           = [];
            hiddenSize          = NumHidden;
            rememberHiddenState = Stateful;
            returnSequence      = ReturnSequence;
            activation          = Activation;
            recurrentActivation = RecurrentActivation;
            resetGateMode       = ResetGateMode;
            
            internalLayer       = nnet.internal.cnn.layer.GRU(name, inputSize, hiddenSize, ...
                rememberHiddenState, returnSequence, activation, recurrentActivation, resetGateMode);
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
                internalLayer.HiddenState                   = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
                internalLayer.HiddenState.Value             = zeros(NumHidden, 1, 'single');
                internalLayer.InitialHiddenState            = zeros(NumHidden, 1, 'single');
            end
            NNTLayers = { nnet.cnn.layer.GRULayer(internalLayer) };
        end
        
        function [tf, Message] = canSupportSettings(~, LSpec)
            % Check unsupported options:
            if ~isempty(kerasField(LSpec, 'activity_regularizer'))
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoActivityRegularization', LSpec.Name);
            elseif logical(kerasField(LSpec, 'go_backwards'))
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:GRUGoBackwards');
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
% The weighting scheme of CuDNNGRU on Keras is different from regular GRU
% weighting scheme, which needs tranpose and reshape so that it can be fit
% in our implementation.

% reorder bias and recurrent bias with DLT order [2 1 3]
 function Bm = reshapeBias(Bk)
    h = size(Bk,1)/6;
    Bm = [Bk(h+1:2*h); Bk(1:h); Bk(2*h+1:3*h); ...
        Bk(4*h+1:5*h); Bk(3*h+1:4*h); Bk(5*h+1:6*h)];
end

%  transpose (and reshape) input and recurrent kernels and reorder
%  according to DLT order [2 1 3]
function Wm = reshapeInputWeight(Wk)
    h = size(Wk,1)/3;
    i = size(Wk,2);
    Wm = [reshape(Wk(h+1:2*h,:), [i, h]), reshape(Wk(1:h,:), [i, h]),...
        reshape(Wk(2*h+1:3*h,:), [i, h])]';
end

function Wm = reshapeRecurrentWeight(Wk)
    h = size(Wk,1)/3;
    assert(h==floor(h));
    Wm = [Wk(h+1:2*h,:), Wk(1:h,:), Wk(2*h+1:3*h,:)]';
end