classdef KerasLayerInsideTimeDistributedModel < nnet.internal.cnn.keras.KerasLayerInsideModel
   
    % Copyright 2019-2023 The Mathworks, Inc.
    
    
    methods
        function this = KerasLayerInsideTimeDistributedModel(Struct, isTensorFlowModel)
            % Struct = 
            %   struct with fields:
            % 
            %     class_name: 'Sequential'
            %         config: [1×1 struct]
            this.Name = Struct.config.name;
            this.ClassName = Struct.class_name;
            this.IsTimeDistributed = true;
            % Note that there shouldn't be another TimeDistributedModel
            % inside of a TimeDistributedModel.
            switch this.ClassName
                case 'Sequential'
                    this.Config = nnet.internal.cnn.keras.KerasSequentialModelConfig(Struct.config, isTensorFlowModel);
                    this.Config.IsTimeDistributed = true;
                case 'Model'
                    this.Config = nnet.internal.cnn.keras.KerasDAGModelConfig(Struct.config, isTensorFlowModel);
                    this.Config.IsTimeDistributed = true;
                otherwise
                    this.Config = nnet.internal.cnn.keras.KerasBaseLevelLayerConfig(Struct.config);
            end
            assert(isfield(Struct, 'inbound_nodes'));
            if ~isempty(Struct.inbound_nodes)
                Node = Struct.inbound_nodes{1};
                this.InboundConnections = cellfun(@nnet.internal.cnn.keras.KerasInputTensorSpec, num2cell(1:numel(Node))', Node, 'UniformOutput', false);
            end
        end
        
        function [LayerSpecs, NameTable] = expandLayer(this, SubmodelName, TimeDistributedName, isTensorFlowModel)
        % layer is a KerasLayerInsideSequentialModel
        switch this.ClassName
            case {'Sequential', 'Model'}
                [LayerSpecs, NameTable] = expandSubmodel(this, SubmodelName, isTensorFlowModel);
            otherwise
                % It's a base layer. No expansion.
                LSpec = nnet.internal.cnn.keras.LayerSpec.fromTimeDistributedBaseLayer(this, SubmodelName, TimeDistributedName, isTensorFlowModel);
                LSpec.InConns = cellfun(@nnet.internal.cnn.keras.util.connFromTensorSpec, this.InboundConnections, 'UniformOutput', false);
                LayerSpecs = {LSpec};
                NameTable = table;
        end
        end
        
        function this = connectSubmodelInbound(this)
            this.Config.Layers{1}.InboundConnections = this.InboundConnections;
        end
    end
end
