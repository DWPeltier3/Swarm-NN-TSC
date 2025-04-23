classdef TranslatorForGRULayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2020 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {'kernel', 'bias', 'recurrent_kernel'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            
            LayerName               = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumHidden               = kerasField(LSpec, 'units');
            ReturnSequence          = logical(kerasField(LSpec, 'return_sequences'));
            Stateful                = true;
            UseBias                 = logical(kerasField(LSpec, 'use_bias'));
            numBiases = 1;
            if isempty(UseBias)
                UseBias = false;    % Empty means false for this field.
            elseif UseBias && TranslateWeights
                numBiases = size(LSpec.Weights.bias, 2);
            end
            Activation              = mapActivation(kerasField(LSpec, 'activation'));
            RecurrentActivation     = mapRecurrentActivation(kerasField(LSpec, 'recurrent_activation'));
            ResetGateMode           = mapResetGateMode(kerasField(LSpec, 'reset_after'), numBiases );
            
            % Create the GRU using an internal layer:
            name = LayerName;
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
            internalLayer.InputWeights.LearnRateFactor 	= single(1);
            internalLayer.InputWeights.L2Factor      	= single(1);
            if TranslateWeights
                verifyWeights(LSpec, 'kernel');
                verifyWeights(LSpec, 'recurrent_kernel');
                internalLayer.InputWeights                	= nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
                internalLayer.InputWeights.Value            = single(reorderGRUWeights(LSpec.Weights.kernel));
                internalLayer.InputWeights.LearnRateFactor 	= single(1);
                internalLayer.InputWeights.L2Factor      	= single(1);
                internalLayer.RecurrentWeights              = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
                internalLayer.RecurrentWeights.Value        = single(reorderGRUWeights(LSpec.Weights.recurrent_kernel));
                internalLayer.RecurrentWeights.LearnRateFactor  = single(1);
                internalLayer.RecurrentWeights.L2Factor     = single(1);
                if UseBias
                    verifyWeights(LSpec, 'bias');
                    internalLayer.Bias.Value                = single(reorderGRUBias(LSpec.Weights.bias));
                else
                    % if UseBias is false, then numBiases = 1
                    internalLayer.Bias.Value                = zeros(3*NumHidden, 1, 'single');
                end
               % internalLayer.Bias.L2Factor                	= single(1);

                internalLayer.HiddenState                   = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
                internalLayer.HiddenState.Value             = zeros(NumHidden, 1, 'single');
                internalLayer.InitialHiddenState            = zeros(NumHidden, 1, 'single');
            end
            NNTLayers = { nnet.cnn.layer.GRULayer(internalLayer) };
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
                Message = message('nnet_cnn_kerasimporter:keras_importer:GRUActivation', kerasField(LSpec, 'activation'));
            elseif isempty(RAName)
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:GRURecurrentActivation', kerasField(LSpec, 'recurrent_activation'));
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

function ResetGateMode = mapResetGateMode(KerasReset_after, numBiases)
if KerasReset_after
    if numBiases == 2
        ResetGateMode = 'recurrent-bias-after-multiplication';
    else
        ResetGateMode = 'after-multiplication';
    end
else
    ResetGateMode = 'before-multiplication';
end
end

function Wm = reorderGRUWeights(Wk)
% GRU weights in Keras are stored in the order:
%     [Update, Reset, Hidden], 
% while in DLT the ordering is:
%     [Reset, Update, Hidden].
idxVec = kerasToDLT(Wk);
Wm = Wk(idxVec,:);
end

function bm = reorderGRUBias(bk)
idxVec = kerasToDLT(bk);
bm = bk(idxVec, :);
bm = bm(:);
end

function idxVec = kerasToDLT(Wk)
h = size(Wk,1)/3;
assert(h==floor(h));
idxVec = [h+1:2*h, 1:h, 2*h+1:3*h]';  % The new block order is [2 1 3].
end