classdef TranslatorForConcatenateLayer < nnet.internal.cnn.keras.LayerTranslator
    
    % Copyright 2017-2019 The Mathworks, Inc.
    
    properties(Constant)
        WeightNames = {};
    end
    
    methods
        function NNTLayers = translate(this, LSpec, TranslateWeights, TranslateTrainingParams, UserImageInputSize)
            % LayerSpec.KerasConfig:
            %   struct with fields:
            %     trainable: 1
            %          name: 'concatenate_1'
            %          axis: -1
            LayerName = nnet.internal.cnn.keras.makeNNTName(LSpec.Name);
            NumInputs = numel(LSpec.InConns);
            ConcatenationAxis = kerasField(LSpec, 'axis');
            
            % TODO : include depthConcatenationLayer when it supports 3D input
            if (ismember(ConcatenationAxis, [-1 3]) && ~LSpec.IsInput3D && ~LSpec.IsFeatureInput) % || (ismember(ConcatenationAxis, [-1 4]) && LSpec.IsInput3D)
                %                 % for 2D data both axis = -1 or 3 is valid
                %                 % for 3D data both axis = -1 or 4 is valid
                NNTLayers = { depthConcatenationLayer(NumInputs, 'Name', LayerName) };
            elseif (ismember(ConcatenationAxis,[1 -1]) && LSpec.IsFeatureInput)
                %for CB input Concat dim = 1
                NNTLayers = { concatenationLayer(1, NumInputs, 'Name', LayerName) }; 
            else
                
                NNTLayers = { concatenationLayer(ConcatenationAxis, NumInputs, 'Name', LayerName) };
            end
        end
        
    end
end