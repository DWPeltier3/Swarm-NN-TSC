classdef TranslatorForDepthwiseConv2DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2018 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {'depthwise_kernel', 'bias'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, ~, ~)
            % Only fields with * are translated. Others are ignored.
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %
            %          depth_multiplier: 1            (*)
            %     depthwise_regularizer: []
            %     depthwise_initializer: [1×1 struct]
            %          bias_initializer: [1×1 struct]
            %                   strides: [2×1 double] (*)
            %               kernel_size: [2×1 double] (*)
            %                  use_bias: 0            (*)
            %      activity_regularizer: []
            %             dilation_rate: [2×1 double] (*)
            %                 trainable: 1 
            %          bias_regularizer: []
            %                activation: 'linear'     (*)
            %           bias_constraint: []
            %                   padding: 'same'       (*)
            %      depthwise_constraint: []
            %               data_format: 'channels_last'
            %                      name: 'expanded_conv_depthwise' (*)
            LayerName = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumFiltersPerGroup = kerasField(LSpec, 'depth_multiplier');
            Stride        = fixKeras2DSizeParameter(this, kerasField(LSpec, 'strides'));
            FilterSize    = fixKeras2DSizeParameter(this, kerasField(LSpec, 'kernel_size'));
            DilationFactor= fixKeras2DSizeParameter(this, kerasField(LSpec, 'dilation_rate'));
            UseBias       = logical(kerasField(LSpec, 'use_bias'));
            if isempty(UseBias)
                UseBias = false;    % Empty means false for this field.
            end
            % padding:
            switch kerasField(LSpec, 'padding')
                case 'valid'
                    Padding = 0;
                case 'same'
                    Padding = 'same';
                otherwise
                    assert(false);
            end
            % Create layer
            layer = groupedConvolution2dLayer(...
                FilterSize,...
                NumFiltersPerGroup,...
                'channel-wise',...
                'Stride', Stride,...
                'DilationFactor', DilationFactor,...
                'Padding', Padding,...
                'Name', LayerName,...
                'BiasLearnRateFactor', single(UseBias));
            
            if TranslateWeights
                % Format of LayerSpec.Weights.depthwise_kernel, and matlab weights:
                % Keras:  depth_multiplier x in_channels x filterWidth x filterHeight
                % MATLAB: filterHeight x filterWidth x NumChannelsPerGroup x
                %         NumFiltersPerGroups x NumGroups.
                verifyWeights(LSpec, 'depthwise_kernel');
                NumGroups = size(LSpec.Weights.depthwise_kernel, 2); 
                NumChannelsPerGroup = 1;
                layer.Weights = single(reshape(...
                    permute(LSpec.Weights.depthwise_kernel, [4 3 1 2]), ...
                    [FilterSize NumChannelsPerGroup NumFiltersPerGroup NumGroups]));
                if UseBias
                    % keras shape: [numberofFilters x 1]. 
                    % Matlab size 1-by-1-by-NumFiltersPerGroup-by-NumGroups.
                    verifyWeights(LSpec, 'bias');
                    layer.Bias = single(reshape(LSpec.Weights.bias, ...
                        [1,1,NumFiltersPerGroup,NumGroups]));
                else
                    layer.Bias = zeros(1,1,NumFiltersPerGroup,NumGroups, 'single');
                end
            end
            NNTLayers = {layer};
            NNTLayers = maybeAppendActivationLayer(LSpec, NNTLayers);
        end
        
        function [tf, Message] = canSupportSettings(~, LSpec)
            [tf, Message] = nnet.internal.cnn.keras.util.canSupportSettingsConv(LSpec); 
            
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
