classdef KerasSequentialModelConfig < nnet.internal.cnn.keras.KerasModelConfig
    
    % Copyright 2019-2023 The Mathworks, Inc.
    
    properties
       InputShape % Store the batch_input_shape/build_input_shape of the model
    end
    
    methods
        function this = KerasSequentialModelConfig(Struct, isTensorFlowModel)
            % Struct = 
            % 
            %   struct with fields:
            % 
            %     layers: [2◊1 struct]
            %       name: 'sequential_2'
            % OR
            % Struct = 
            % 
            %   4◊1 struct array with fields:
            % 
            %     class_name
            %     config
            [layers, this.Name] = nnet.internal.cnn.keras.getLayersFromSequentialModel(Struct);
            % set inbound_nodes to layer structs
            layers = iSetInboundNodes(layers);

            numLayers = ones(numel(layers), 1);
            this.Layers = arrayfun(@nnet.internal.cnn.keras.KerasLayerInsideSequentialModel, layers, logical(numLayers .* isTensorFlowModel), 'UniformOutput', false);

            input_layers = iSetInputLayer(this.Layers{1});
            this.InputLayers = cellfun(@nnet.internal.cnn.keras.KerasInputTensorSpec, num2cell(1:numel(input_layers))', input_layers, 'UniformOutput', false);
            output_layers = iSetOutputLayer(this.Layers{end});
            this.OutputLayers = cellfun(@nnet.internal.cnn.keras.KerasOutputTensorSpec, num2cell(1:numel(output_layers))', output_layers, 'UniformOutput', false);
            this = setInputShape(this, Struct);
            this = setFirstLayerInputShape(this);
        end
        
        function this = setInputShape(this, Struct)
            if isfield(Struct, 'build_input_shape')
               this.InputShape = Struct.build_input_shape; 
            end
        end
        
        function this = setFirstLayerInputShape(this)
            if ~isempty(this.InputShape)
                if isa(this.Layers{1}.Config, 'nnet.internal.cnn.keras.KerasBaseLevelLayerConfig')
                    this.Layers{1}.Config.KerasStruct.batch_input_shape = this.InputShape;
                elseif isa(this.Layers{1}.Config, 'nnet.internal.cnn.keras.KerasSequentialModelConfig')
                    this.Layers{1}.Config.InputShape = this.InputShape;
                end
            end
        end
    end
end

function layers = iSetInboundNodes(layers)
% TODO: change inbound_nodes{2} when supporting weight sharing
%     layerReplica = containers.Map();
    for L = 2:numel(layers)
        inbound_nodes = cell(4, 1);
        inbound_nodes{1} = layers(L-1).config.name;
        inbound_nodes{3} = 0;
        inbound_nodes{4} = struct();
        layers(L).inbound_nodes = {{inbound_nodes}};
    end
end

function inputLayerSpec = iSetInputLayer(inputLayer)
% TODO: change inputLayerSpecs{2} when supporting weight sharing
    inputLayerSpec = cell(4, 1);
    inputLayerSpec{1} = inputLayer.Name;
    inputLayerSpec{3} = 0;
    inputLayerSpec{4} = struct();
    inputLayerSpec = {inputLayerSpec};
end

function outputLayerSpec = iSetOutputLayer(outputLayer)
    outputLayerSpec = cell(3, 1);
    outputLayerSpec{1} = outputLayer.Name;
    outputLayerSpec{3} = 0; 
    outputLayerSpec = {outputLayerSpec};
end
