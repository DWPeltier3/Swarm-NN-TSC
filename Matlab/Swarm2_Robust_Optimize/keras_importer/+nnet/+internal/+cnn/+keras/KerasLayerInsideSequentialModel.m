classdef KerasLayerInsideSequentialModel < nnet.internal.cnn.keras.KerasLayerInsideModel

% Copyright 2017-2023 The MathWorks, Inc.

    methods
        function this = KerasLayerInsideSequentialModel(Struct, isTensorFlow)
            % Struct =
            %   struct with fields:
            %         config: [1◊1 struct]
            %     class_name: 'Conv2D'
            this.Name = Struct.config.name;
            this.ClassName = Struct.class_name;
            switch this.ClassName
                case 'Sequential'
                    this.Config = nnet.internal.cnn.keras.KerasSequentialModelConfig(Struct.config, isTensorFlow);
                case 'Model'
                    this.Config = nnet.internal.cnn.keras.KerasDAGModelConfig(Struct.config, isTensorFlow);
                case 'TimeDistributed'
                    if ~isTensorFlow || nnet.internal.cnn.keras.util.checkSupportsTimeDistributedInDlnetwork(Struct.config.layer)
                        this.Config = nnet.internal.cnn.keras.KerasTimeDistributedModelConfig(Struct.config, isTensorFlow);
                    else
                        % Assume it's a base layer
                        this.Config = nnet.internal.cnn.keras.KerasBaseLevelLayerConfig(Struct.config);
                        if isfield(Struct, 'inbound_nodes') 
                            this.Config.KerasStruct.inbound_nodes = Struct.inbound_nodes;
                        end
                    end
                otherwise
                    this.Config = nnet.internal.cnn.keras.KerasBaseLevelLayerConfig(Struct.config);
                    if isfield(Struct, 'inbound_nodes') 
                        this.Config.KerasStruct.inbound_nodes = Struct.inbound_nodes;
                    end
            end
            if isfield(Struct, 'inbound_nodes') && ~isempty(Struct.inbound_nodes)
                Node = Struct.inbound_nodes{1};
                this.InboundConnections = cellfun(@nnet.internal.cnn.keras.KerasInputTensorSpec, num2cell(1:numel(Node))', Node, 'UniformOutput', false);
            end
        end
        
        function this = connectSubmodelInbound(this)
            this.Config.Layers{1}.InboundConnections = this.InboundConnections;
        end
    end
end