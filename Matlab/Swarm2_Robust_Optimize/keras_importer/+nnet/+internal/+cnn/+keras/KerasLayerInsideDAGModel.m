classdef KerasLayerInsideDAGModel < nnet.internal.cnn.keras.KerasLayerInsideModel

% Copyright 2017-2023 The MathWorks, Inc.

    methods
        function this = KerasLayerInsideDAGModel(Struct, isTensorFlowModel)
            % Struct =
            %   struct with fields:
            %
            %        class_name: 'InputLayer'
            %            config: [1�1 struct]
            %              name: 'input_1'
            %     inbound_nodes: []
            this.Name = Struct.name;
            this.ClassName = Struct.class_name; 
            switch this.ClassName
                case 'Model'
                    this.Config = nnet.internal.cnn.keras.KerasDAGModelConfig(Struct.config, isTensorFlowModel);
                case 'Sequential'
                    this.Config = nnet.internal.cnn.keras.KerasSequentialModelConfig(Struct.config, isTensorFlowModel);
                case 'TimeDistributed'
                    if ~isTensorFlowModel || nnet.internal.cnn.keras.util.checkSupportsTimeDistributedInDlnetwork(Struct.config.layer)
                        this.Config = nnet.internal.cnn.keras.KerasTimeDistributedModelConfig(Struct.config, isTensorFlowModel);
                    else
                        % Assume it's a base layer
                        this.Config = nnet.internal.cnn.keras.KerasBaseLevelLayerConfig(Struct.config);
                        if isfield(Struct, 'inbound_nodes') 
                            this.Config.KerasStruct.inbound_nodes = Struct.inbound_nodes;
                        end
                    end
                otherwise
                    % Assume it's a base layer
                    this.Config = nnet.internal.cnn.keras.KerasBaseLevelLayerConfig(Struct.config);
                    if isfield(Struct, 'inbound_nodes') 
                        this.Config.KerasStruct.inbound_nodes = Struct.inbound_nodes;
                    end
            end
            % Process inbound nodes
            if isstruct(Struct.inbound_nodes) 
                Struct.inbound_nodes = struct2cell(Struct.inbound_nodes); 
            end 
            if numel(Struct.inbound_nodes) > 1
                % This is the case of weight sharing
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:NoWeightSharing')));
            else
                if ~isempty(Struct.inbound_nodes)
                    Node = Struct.inbound_nodes{1};
                    if strcmp(this.ClassName,'TFOpLambda') && numel(Node) == 1 && numel(Node{1}) > 3 && isstruct(Node{1}{4})
                        % Process nested inbound connections for TFOpLambda functions
                        nestedStruct = Node{1}{4};
                        Node{1}(4) = [];
                        fns = fieldnames(nestedStruct);
                        for i = 1:numel(fns)
                            if numel(nestedStruct.(fns{i})) == 3
                                % Potential inbound connection from a previous layer                                
                                Node{end+1} = nestedStruct.(fns{i}); %#ok<AGROW>
                            end
                        end
                        Node = Node';
                    end

                    if strcmp(this.ClassName,'TFOpLambda') && numel(Node) > 1
                        % Remove Constant value inbound connections
                        inboundNodes = Node;
                        for i = 1:numel(inboundNodes)
                           if strcmp(inboundNodes{i}{1}, '_CONSTANT_VALUE')
                                Node(i) = [];
                           end
                        end
                    end
                    this.InboundConnections = cellfun(@nnet.internal.cnn.keras.KerasInputTensorSpec, num2cell(1:numel(Node))', Node, 'UniformOutput', false); 
                end
            end
        end
        
        function this = connectSubmodelInbound(this)
            switch this.ClassName
                case 'Model'
                    this = changeLayerInboundToDAGInbound(this);
                case 'Sequential'
                    % This layer is a Sequential Layer, let the first layer's
                    % InboundConnections to be the InboundConnection of the
                    % sequential model.
                    this.Config.Layers{1}.InboundConnections = this.InboundConnections;
                case 'TimeDistributed'
                    this.Config.Layers{1}.InboundConnections = this.InboundConnections;
            end
        end
        
    end
    
    methods(Access=private)
        function this = changeLayerInboundToDAGInbound(this)
            % This layer is a DAG model, which will contain sublayers that read its
            % internal inputs, which are effectively dummy inputs that will not be used
            % by the layers of this model. Change the InboundConnections of those
            % sublayers to the InboundConnections of the parent layer. Alg: For each
            % element of layer.Config.InputLayers, find all sublayers that refer to it
            % in their InboundConnections, and replace those references with the
            % corresponding InboundConnection of 'layer.InboundConnections'.
            for LayerConnNum = 1:numel(this.Config.InputLayers)
                for SublayerNum = 1:numel(this.Config.Layers)
                    for SubConnNum = 1:numel(this.Config.Layers{SublayerNum}.InboundConnections)
                        if isequal(this.Config.Layers{SublayerNum}.InboundConnections{SubConnNum}, this.Config.InputLayers{LayerConnNum})
                            this.Config.Layers{SublayerNum}.InboundConnections{SubConnNum} = this.InboundConnections{LayerConnNum};
                        end
                    end
                end
            end
        end
        % Add function here to check if layer is TimeDistributedCompatible
        % or can add it in the base class
    end
end