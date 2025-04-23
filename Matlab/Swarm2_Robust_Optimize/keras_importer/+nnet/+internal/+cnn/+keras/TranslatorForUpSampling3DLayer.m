classdef TranslatorForUpSampling3DLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2020-2022 The MathWorks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % This layer increases the resolution of 3D inputs
            % LayerConfig =
            %   struct with fields:
            %
            %     data_format: 'channels_last'
            %       trainable: 1
            %            size: [2x1 double]
            %            name: 'upsampling3d_1'
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);

            % UpSampling3D layer only takes integer values for the size
            % parameter in TensorFlow.
            Scale       = fixKeras3DSizeParameter(this, kerasField(LSpec, 'size'));
            NNTLayers = { resize3dLayer('Scale', Scale, 'Name', LayerName) };
        end
    end
end