classdef TranslatorForTimeDistributedLayerIn < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2019 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            
            NNTLayers = { sequenceFoldingLayer('Name', nnet.internal.cnn.keras.makeNNTName(LSpec.Name)) };
        end
    end
end