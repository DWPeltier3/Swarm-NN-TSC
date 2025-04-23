classdef TranslatorForConv1DLayer < nnet.internal.cnn.keras.LayerTranslator
    % Copyright 2021 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {'kernel', 'bias'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
%             LSpec.KerasConfig
%             ans = 
%               struct with fields:
%             
%                                 name: 'conv1d_1'
%                            trainable: 1
%                    batch_input_shape: [3×1 double]
%                                dtype: 'float32'
%                              filters: 64
%                          kernel_size: 3
%                              strides: 1
%                              padding: 'valid'
%                          data_format: 'channels_last'
%                        dilation_rate: 1
%                           activation: 'relu'
%                             use_bias: 1
%                   kernel_initializer: [1×1 struct]
%                     bias_initializer: [1×1 struct]
%                   kernel_regularizer: []
%                     bias_regularizer: []
%                 activity_regularizer: []
%                    kernel_constraint: []
%                      bias_constraint: []
            LayerName     = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumFilters    = kerasField(LSpec, 'filters');
            FilterSize    = kerasField(LSpec, 'kernel_size');
            Stride        = kerasField(LSpec, 'strides');
            DilationFactor= kerasField(LSpec, 'dilation_rate');
            UseBias       = logical(kerasField(LSpec, 'use_bias'));
            if isempty(UseBias)
                UseBias = false;    % Empty means false for this field.
            end
            % padding:
            switch kerasField(LSpec, 'padding')
                case 'valid'
                    Padding = [0 0];
                case 'same'
                    Padding = 'same';
                case 'causal'
                    Padding = 'causal';
                otherwise
                    assert(false);
            end
            % Create layer
            conv1d = convolution1dLayer(FilterSize, NumFilters,...
                'Stride', Stride,...
                'DilationFactor', DilationFactor,...
                'Padding', Padding,...
                'Name', LayerName,...
                'BiasLearnRateFactor', single(UseBias));
            if TranslateWeights
                if UseBias
                    % keras shape: [numberofFilters x 1]. Matlab size 1-by-NumFilters.
                    verifyWeights(LSpec, 'bias');
                    conv1d.Bias = single(reshape(LSpec.Weights.bias, [1,NumFilters]));
                else
                    conv1d.Bias = zeros(1,NumFilters, 'single');
                end
                % Format of LayerSpec.Weights.kernel, and matlab conv1d weights:
                % Keras:  NumFilters-NumChannels-FilterSize:    F x C x L
                % MATLAB: FilterSize-NumChannels-NumFilters:  L x C x F
                verifyWeights(LSpec, 'kernel');
                conv1d.Weights = single(permute(LSpec.Weights.kernel, [3,2,1]));
            end
            NNTLayers = {conv1d};
            NNTLayers = maybeAppendActivationLayer(LSpec, NNTLayers);
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if ~ismember(kerasField(LSpec, 'padding'), {'valid', 'same','causal'})
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedPadding', LSpec.Name);
            elseif ~isempty(kerasField(LSpec, 'activity_regularizer'))
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoActivityRegularization', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
            
            if LSpec.isTensorFlowLayer
            % Do not generate a layer if the activation is not supported for TF Layers.
                supportedActivations = nnet.internal.cnn.keras.getSupportedActivations(); 
                if hasKerasField(LSpec, 'activation') && ~ismember(kerasField(LSpec, 'activation'), supportedActivations)
                    tf = false; 
                    Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedActivation', ...
                        kerasField(LSpec, 'activation'));
                end
            end
        end
    end
end

