classdef TranslatorForUnsupportedKerasLayer < nnet.internal.cnn.keras.LayerTranslator

% Copyright 2017-2018 The Mathworks, Inc.

    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig
            % ans =
            %   struct with fields:
            %          name: 'xxxx'
            %           < other fields ignored >
            %
            LayerName   = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            numInputs   = max(1, numel(LSpec.InConns));
            numOutputs  = LSpec.NumOutputs;
            Layer       = nnet.keras.layer.PlaceholderLayer(LayerName, this.KerasLayerType, ...
                                LSpec.KerasConfig, numInputs, numOutputs);
            if TranslateWeights
                Layer.Weights = LSpec.Weights;
            end
            NNTLayers = { Layer };
        end
    end
end

