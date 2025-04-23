classdef TranslatorForConv2DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017-2018 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {'kernel', 'bias'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %              kernel_size: [2x1 double]
            %         bias_initializer: [1x1 struct]
            %        kernel_constraint: []
            %                  strides: [2x1 double]
            %                 use_bias: 1
            %         bias_regularizer: []
            %       kernel_regularizer: []
            %     activity_regularizer: []
            %       kernel_initializer: [1x1 struct]
            %                  filters: 8
            %              data_format: 'channels_last'
            %                     name: 'conv2d_1'
            %          bias_constraint: []
            %                trainable: 1
            %                  padding: 'valid'
            %               activation: 'linear'
            %            dilation_rate: [2x1 double]
            LayerName     = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumFilters    = kerasField(LSpec, 'filters');
            FilterSize    = fixKeras2DSizeParameter(this, kerasField(LSpec, 'kernel_size'));
            Stride        = fixKeras2DSizeParameter(this, kerasField(LSpec, 'strides'));
            DilationFactor= fixKeras2DSizeParameter(this, kerasField(LSpec, 'dilation_rate'));
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
                otherwise
                    assert(false);
            end
            % Create layer
            conv2d = convolution2dLayer(FilterSize, NumFilters,...
                'Stride', Stride,...
                'DilationFactor', DilationFactor,...
                'Padding', Padding,...
                'Name', LayerName,...
                'BiasLearnRateFactor', single(UseBias));
            if TranslateWeights
                if UseBias
                    % keras shape: [numberofFilters x 1]. Matlab size 1-by-1-by-NumFilters.
                    verifyWeights(LSpec, 'bias');
                    conv2d.Bias = single(reshape(LSpec.Weights.bias, [1,1,NumFilters]));
                else
                    conv2d.Bias = zeros(1,1,NumFilters, 'single');
                end
                % Format of LayerSpec.Weights.kernel, and matlab conv2d weights:
                % Keras:  NumFilters-NumChannels-ncol-nRow:    F x C x W x H
                % MATLAB: Height-Width-NumChannels-NumFilters: H x W x C x F
                verifyWeights(LSpec, 'kernel');
                conv2d.Weights = single(permute(LSpec.Weights.kernel, [4,3,2,1]));
            end
            NNTLayers = {conv2d};
            NNTLayers = maybeAppendActivationLayer(LSpec, NNTLayers);
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if ~ismember(kerasField(LSpec, 'padding'), {'valid', 'same'})
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
