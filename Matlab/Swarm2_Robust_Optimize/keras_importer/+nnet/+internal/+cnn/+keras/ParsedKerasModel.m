classdef ParsedKerasModel < handle

% Copyright 2019 The Mathworks, Inc.

    properties
        ClassName
        Model          % Either KerasSequentialModel or a KerasDAGModel
        TrainingConfig        
    end
    
    methods
        function this = ParsedKerasModel(KerasModelConfig, TrainingConfig, isTFModel) 
            % Parse keras structs into an internal representation.
            % KerasModelConfig =
            %   struct with fields:
            %         config: [7×1 struct]
            %     class_name: 'Sequential'
            this.ClassName = KerasModelConfig.class_name;
            this.TrainingConfig = TrainingConfig;
            if nargin == 2
                isTFModel = false;
            end
            
            switch this.ClassName
                case 'Sequential'
                    this.Model = nnet.internal.cnn.keras.KerasSequentialModel(KerasModelConfig, isTFModel);
                case {'Model', 'Functional'}
                    this.Model = nnet.internal.cnn.keras.KerasDAGModel(KerasModelConfig, isTFModel);
                otherwise
                    assert(false);
            end
            visited = checkWeightSharing(this.Model.Config, {});
        end
    end
end
