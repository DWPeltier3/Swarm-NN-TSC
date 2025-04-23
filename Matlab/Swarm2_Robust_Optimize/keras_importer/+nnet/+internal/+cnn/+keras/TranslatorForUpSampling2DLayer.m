classdef TranslatorForUpSampling2DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2020-2022 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % This layer increases the resolution of 2D inputs
            % LayerConfig =
            %   struct with fields:
            %
            %     data_format: 'channels_last'
            %       trainable: 1
            %            size: [2x1 double]
            %            name: 'upsampling2d_1'
            %       interpolation: 'nearest'
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);

            % UpSampling2D layer takes integer values for the size
            % parameter in TensorFlow. However, if given a float value, it 
            % applies the floor operation internally and creates the layer 
            % using the rounded numbers. Hence, a floor operation is added 
            % here to get similar behavior as in TensorFlow.
            Scale = floor(fixKeras2DSizeParameter(this, kerasField(LSpec, 'size')));

            % For older versions where inpterpolation method field does not 
            % exist, will use the default method which is 'nearest'. Other 
            % interpolation methods from the keras UpSampling2D layer are 
            % not supported currently. Will default to 'bilinear' for 
            % those methods.
            if ~hasKerasField(LSpec, 'interpolation')
                NNTLayers = { resize2dLayer('Scale', Scale, 'Method', 'nearest', 'Name', LayerName) };
            elseif ismember(kerasField(LSpec,'interpolation'), ["bilinear", "nearest"])
                Method = kerasField(LSpec, 'interpolation');
                NNTLayers = { resize2dLayer('Scale', Scale, 'Method', Method, 'Name', LayerName) };
            else
                NNTLayers = { resize2dLayer('Scale', Scale, 'Method', 'bilinear', 'Name', LayerName) };
                iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:UnsupportedInterpolationMethod', LSpec.Name);
            end
        end
    end
end

function iWarningWithoutBacktrace(msgID, varargin)
nnet.internal.cnn.keras.util.warningWithoutBacktrace(msgID, varargin{:})
end