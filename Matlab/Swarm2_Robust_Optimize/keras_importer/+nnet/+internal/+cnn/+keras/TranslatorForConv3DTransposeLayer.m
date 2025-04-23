classdef TranslatorForConv3DTransposeLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017-2019 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {'kernel', 'bias'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            %   LayerSpec.KerasConfig:
            % struct with fields:
            %
            %       kernel_regularizer: []
            %          bias_constraint: []
            %         bias_initializer: [1x1 struct]
            %                 use_bias: 0
            %                trainable: 1
            %                  strides: [3x1 double]
            %                  padding: 'valid'
            %                  filters: 59
            %                     name: 'conv3d_transpose_1'
            %       kernel_initializer: [1x1 struct]
            %              data_format: 'channels_last'
            %     activity_regularizer: []
            %               activation: 'relu'
            %        kernel_constraint: []
            %         bias_regularizer: []
            %              kernel_size: [3x1 double]
            %------------------------------------------------------
            % Keras 2.2.3 added dilation_rate argument
            %
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumFilters  = kerasField(LSpec, 'filters');
            FilterSize  = fixKeras3DSizeParameter(this, kerasField(LSpec, 'kernel_size'));
            Stride      = fixKeras3DSizeParameter(this, kerasField(LSpec, 'strides'));
            UseBias     = logical(kerasField(LSpec, 'use_bias'));
            if isempty(UseBias)
                UseBias = false;    % Empty means false for this field.
            end
            switch kerasField(LSpec, 'padding')
                case 'valid'
                    Cropping = [0 0 0;0 0 0];
                case 'same'
                    Cropping = 'same';
                otherwise
                    assert(false);
            end
            % Create layer
            Conv3dTransposed = transposedConv3dLayer(FilterSize, NumFilters,...
                'Stride', Stride,...
                'Cropping', Cropping,...
                'Name', LayerName,...
                'BiasLearnRateFactor', single(UseBias));
            if TranslateWeights
                % keras shape: [numberofFiltersx1], Maltab size 1-by-1-by-1-by-NumFilters matrix.
                if UseBias
                    verifyWeights(LSpec, 'bias');
                    Conv3dTransposed.Bias = single(reshape(LSpec.Weights.bias, [1,1,1,NumFilters]));
                else
                    Conv3dTransposed.Bias = zeros(1,1,1,NumFilters, 'single');
                end
                % format of LayerSpec.Weights.kernel: NumChannels-NumFilters-depth-ncol-nRow:
                %                                                       C x F x D x W x H
                % MATLAB: FilterSize(1)-by-FilterSize(2)-by-FilterSize(3)-by-NumFilters-by-NumChannels:
                %                                                       H x W x D x F x C
                verifyWeights(LSpec, 'kernel');
                Conv3dTransposed.Weights = single(permute(LSpec.Weights.kernel, [5,4,3,2,1]));
            end
            NNTLayers = {Conv3dTransposed};
            NNTLayers = maybeAppendActivationLayer(LSpec, NNTLayers);
        end
        
        function [tf, Message] = canSupportSettings(this, LSpec)
            if ~ismember(kerasField(LSpec, 'padding'), {'valid', 'same'})
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedPadding', LSpec.Name);
            elseif ~isempty(kerasField(LSpec, 'activity_regularizer'))
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoActivityRegularization', LSpec.Name);
            elseif hasKerasField(LSpec, 'dilation_rate') && any(kerasField(LSpec, 'dilation_rate') > 1)
                % Optional. Introduced in Keras 2.2.3
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:NoDilationRate', LSpec.Name);
            elseif any(kerasField(LSpec, 'strides') > kerasField(LSpec, 'kernel_size'))
                if hasKerasField(LSpec, 'output_padding') && ...
                        ~isempty(kerasField(LSpec, 'output_padding')) && ...
                        all(kerasField(LSpec, 'output_padding') == 0)
                    % The layer has output_padding values that are all zero
                    tf = true;
                    Message = '';                   
                else 
                    tf = false;
                    Message = message('nnet_cnn_kerasimporter:keras_importer:StrideGreaterThanKernelSize', LSpec.Name);
                end
            elseif hasKerasField(LSpec, 'output_padding') && any(kerasField(LSpec, 'output_padding') ~= 0)
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:ConvTransposeOutputPadding', LSpec.Name);                
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