classdef TranslatorForBatchNormalizationLayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2017-2022 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {'beta', 'gamma', 'moving_mean', 'moving_variance'};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %                            axis: -1
            %                beta_initializer: [1×1 struct]
            %                       trainable: 1
            %                           dtype: 'float32'
            %                        momentum: 0.9900
            %                         epsilon: 1.0000e-03
            %                           scale: 1
            %                            name: 'batch_normalization_1'
            %                          center: 1
            %     moving_variance_initializer: [1×1 struct]
            %                beta_regularizer: []
            %                 beta_constraint: []
            %                gamma_constraint: []
            %               gamma_initializer: [1×1 struct]
            %               gamma_regularizer: []
            %               batch_input_shape: [4×1 double]
            %         moving_mean_initializer: [1×1 struct]
            LayerName = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            Epsilon   = kerasField(LSpec, 'epsilon');
            DoCenter  = logical(kerasField(LSpec, 'center'));
            DoScale   = logical(kerasField(LSpec, 'scale'));
            if TranslateWeights
                % LayerSpec.Weights
                % ans =
                %   struct with fields:
                %
                %                beta: [32×1 single]
                %               gamma: []
                %         moving_mean: [32×1 single]
                %     moving_variance: [32×1 single]
                NumChannels  	= size(LSpec.Weights.moving_mean, 1);
                verifyWeights(LSpec, 'moving_mean');
                verifyWeights(LSpec, 'moving_variance');

                % Reshaping to the canonical format, i.e. 1x1xC for SSCB is
                % performed in the network assembly.
                TrainedMean     = single(LSpec.Weights.moving_mean);
                TrainedVariance = single(LSpec.Weights.moving_variance);
                if DoCenter
                    verifyWeights(LSpec, 'beta');
                    Offset = single(LSpec.Weights.beta);
                else
                    Offset = zeros([NumChannels,1],'single');
                end

                if DoScale
                    verifyWeights(LSpec, 'gamma');
                    Scale = single(LSpec.Weights.gamma);
                else
                    Scale = ones([NumChannels,1],'single');
                end

                OffsetLearnRateFactor = single(DoCenter);
                OffsetL2Factor = single(1);

                ScaleLearnRateFactor = single(DoScale);
                ScaleL2Factor = single(1);

                % Set nonpositive variance components to a value below eps('single')
                nonPos = TrainedVariance <= 0;
                if any(nonPos)
                    nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:BatchNormNegVar', LayerName)
                    TrainedVariance(nonPos) = realmin('single');
                end

                BN = batchNormalizationLayer('Name', LayerName, 'Offset', Offset,...
                    'Scale', Scale, 'Epsilon', Epsilon, 'OffsetLearnRateFactor', OffsetLearnRateFactor,...
                    'ScaleLearnRateFactor', ScaleLearnRateFactor, 'OffsetL2Factor', OffsetL2Factor,...
                    'ScaleL2Factor', ScaleL2Factor, 'TrainedMean', TrainedMean, ...
                    'TrainedVariance', TrainedVariance);
            else
                BN = batchNormalizationLayer('Name', LayerName, 'Epsilon', Epsilon);
            end
            NNTLayers = {BN};
        end
        
        function [tf, Message] = canSupportSettings(~, LSpec)
            % BatchNormalizationLayer can operate over any data format
            % provided the channel dimension location is known. This is
            % infered *not* from the axis value in Keras but from the
            % format propagation in assembling the network
            if ~(ismember(kerasField(LSpec, 'axis'), [-1 1 2 3 4 5]))
                tf = false;
                Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedAxis', LSpec.Name);
            else
                tf = true;
                Message = '';
            end
        end
    end
end