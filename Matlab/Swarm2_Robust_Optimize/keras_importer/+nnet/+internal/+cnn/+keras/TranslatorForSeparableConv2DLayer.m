classdef TranslatorForSeparableConv2DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2018 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {'depthwise_kernel', 'pointwise_kernel', 'bias'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, ~, ~)
            % Only fields with * are translated. Others are ignored.
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %           
            %           bias_constraint: []
            %                 trainable: 1
            %      pointwise_constraint: []
            %     depthwise_regularizer: []
            %                activation: 'linear'      (*)
            %                  use_bias: 1             (*)
            %         batch_input_shape: [4×1 double]
            %     depthwise_initializer: [1×1 struct]
            %     pointwise_regularizer: []
            %      depthwise_constraint: []
            %                   filters: 32            (*)
            %      activity_regularizer: []
            %                   padding: 'valid'       (*)
            %          bias_initializer: [1×1 struct]
            %             dilation_rate: [2×1 double]  (*)
            %                   strides: [2×1 double]  (*)
            %          depth_multiplier: 1             (*)
            %               data_format: 'channels_last'
            %                      name: 'separable_conv2d_2' (*)
            %               kernel_size: [2×1 double]  (*)
            %     pointwise_initializer: [1×1 struct]
            %          bias_regularizer: []
            %                     dtype: 'float32'
            % -------------------------------------------------------------
            % LayerSpec.Weights
            % ans = 
            %   struct with fields:
            % 
            %     depthwise_kernel: [1×3×3×3 single]
            %     pointwise_kernel: [32×3 single]
            %                 bias: [32×1 single]
            LayerName = string(nnet.internal.cnn.keras.makeNNTName(LSpec.Name));
            ChannelWiseLayerName = LayerName + "_channel-wise";
            PointWiseLayerName = LayerName + "_point-wise";
            
            % Channel-wise step:            
            NumFiltersPerGroup = kerasField(LSpec, 'depth_multiplier');
            Stride        = fixKeras2DSizeParameter(this, kerasField(LSpec, 'strides'));
            FilterSize    = fixKeras2DSizeParameter(this, kerasField(LSpec, 'kernel_size'));
            DilationFactor= fixKeras2DSizeParameter(this, kerasField(LSpec, 'dilation_rate'));
            switch kerasField(LSpec, 'padding')
                case 'valid'
                    Padding = 0;
                case 'same'
                    Padding = 'same';
                otherwise
                    assert(false);
            end
            channelWise = groupedConvolution2dLayer(...
                FilterSize,...
                NumFiltersPerGroup,...
                'channel-wise',...
                'Stride', Stride,...
                'DilationFactor', DilationFactor,...
                'Padding', Padding,...
                'Name', ChannelWiseLayerName,...
                'BiasLearnRateFactor', single(0));

            % Point-wise step:
            NumFilters    = kerasField(LSpec, 'filters');
            UseBias       = logical(kerasField(LSpec, 'use_bias'));
            if isempty(UseBias)
                UseBias = false;    % Empty means false for this field.
            end
            pointWise = convolution2dLayer(...
                1,...
                NumFilters,...
                'Name', PointWiseLayerName,...
                'BiasLearnRateFactor', single(UseBias));
                      
            if TranslateWeights
                % Format of LayerSpec.Weights.depthwise_kernel, and matlab weights:
                % Keras:  depth_multiplier x in_channels x filterWidth x filterHeight
                % MATLAB: filterHeight x filterWidth x NumChannelsPerGroup x
                %         NumFiltersPerGroups x NumGroups.
                verifyWeights(LSpec, 'depthwise_kernel');
                NumGroups = size(LSpec.Weights.depthwise_kernel, 2); 
                NumChannelsPerGroup = 1;
                channelWise.Weights = single(reshape(...
                    permute(LSpec.Weights.depthwise_kernel, [4 3 1 2]), ...
                    [FilterSize NumChannelsPerGroup NumFiltersPerGroup NumGroups]));
                channelWise.Bias = zeros(1,1,NumFiltersPerGroup,NumGroups, 'single');
                % Format of LayerSpec.Weights.pointwise_kernel, and matlab weights:
                % Keras:  filters x in_channels
                % MATLAB: filterHeight x filterWidth x NumChannels x
                %   NumFilters
                verifyWeights(LSpec, 'pointwise_kernel');
                NumChannels = size(LSpec.Weights.pointwise_kernel, 2);
                pointWise.Weights = single(reshape(...
                    permute(LSpec.Weights.pointwise_kernel, [2 1]), ...
                    [1 1 NumChannels NumFilters]));
                if UseBias
                    % keras shape: [numberofFilters x 1]. 
                    % Matlab size 1-by-1-by-NumFilters.
                    verifyWeights(LSpec, 'bias');
                    pointWise.Bias = single(reshape(LSpec.Weights.bias, ...
                        [1,1,NumFilters]));
                else
                    pointWise.Bias = zeros(1,1,NumFilters, 'single');
                end
            end
            NNTLayers = {channelWise, pointWise};
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
