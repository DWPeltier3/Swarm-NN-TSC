classdef KerasTimeDistributedModelConfig < nnet.internal.cnn.keras.KerasModelConfig
    
    % Copyright 2019-2023 The Mathworks, Inc.
    
    methods
        function this = KerasTimeDistributedModelConfig(Struct, isTensorFlowModel)
            % Struct = 
            % 
            %             dtype: 'float32'
            % batch_input_shape: [5×1 double]
            %         trainable: 1
            %              name: 'time_distributed_85'
            %             layer: [1×1 struct]
            this.Name = Struct.name;
            this.IsTimeDistributed = true;
            % Struct.layer is a wrapped layer, which may be a single layer
            % or a sequential model or a DAG model
            batch_input_shape = [];
            if isfield(Struct, 'batch_input_shape')
                batch_input_shape = Struct.batch_input_shape;
            end
            % We are adding fake timeDistributedIn Keras Layer and fake
            % timeDistributedOut Keras Layer here.
            layers = [getTimeDistributedLayer(this, 'in', batch_input_shape); Struct.layer; getTimeDistributedLayer(this, 'out', [])];
            layers = iSetInboundNodes(layers);

            numLayers = ones(numel(layers), 1);
            this.Layers = arrayfun(@nnet.internal.cnn.keras.KerasLayerInsideTimeDistributedModel, layers, logical(numLayers .* isTensorFlowModel), 'UniformOutput', false);

            input_layers = iSetInputLayer(this.Layers{1});
            this.InputLayers = cellfun(@nnet.internal.cnn.keras.KerasInputTensorSpec, num2cell(1:numel(input_layers))', input_layers, 'UniformOutput', false);
            output_layers = iSetOutputLayer(this.Layers{end});
            this.OutputLayers = cellfun(@nnet.internal.cnn.keras.KerasOutputTensorSpec, num2cell(1:numel(output_layers))', output_layers, 'UniformOutput', false);
        end
        
        function [LayerSpecs, NameTable] = flattenLayer(this, SubmodelName, isTFModel)
            % Expand each layer of the Sequential into a flat list of LayerSpecs
            % L is KerasLayerInsideModel
            this.Layers = cellfun(@(L)setfield(L, 'IsTimeDistributed', this.IsTimeDistributed), this.Layers, 'UniformOutput', false);
            [ExpandedLayerSpecs, ExpandedNameTables] = cellfun(@(L)expandLayer(L, SubmodelName, this.Name, isTFModel), ...
            this.Layers, 'UniformOutput', false);
            % Concatenate all the LayerSpec lists and NameTables
            LayerSpecs = unionLayerSpecLists(ExpandedLayerSpecs);
            % LayerSpecs = vertcat(ExpandedLayerSpecs{:});
            NameTable = vertcat(ExpandedNameTables{:});
            % Apply NameTable to the LayerSpecs
            LayerSpecs = applyNameTableToLayers(NameTable, LayerSpecs);
        end
        
        function timeDistributed = getTimeDistributedLayer(this, mode, batch_input_shape)
            % This function is to create fake Keras Layers
            timeDistributed = struct();
            if isequal(mode, 'in')
                timeDistributed.class_name = 'TimeDistributedIn';
            else
                timeDistributed.class_name = 'TimeDistributedOut';
            end
            timeDistributed.config = struct();
            timeDistributed.config.name = [this.Name '_' mode];
            timeDistributed.config.batch_input_shape = batch_input_shape;
        end
    end
end

function layers = iSetInboundNodes(layers)
% change inbound_nodes{2} when supporting weight sharing
    for L = 2:numel(layers)
        inbound_nodes = cell(4, 1);
        inbound_nodes{1} = layers(L-1).config.name;
        inbound_nodes{3} = 0;
        inbound_nodes{4} = struct();
        layers(L).inbound_nodes = {{inbound_nodes}};
    end
end

function inputLayerSpec = iSetInputLayer(inputLayer)
% change inputLayerSpecs{2} when supporting weight sharing
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

function LayerSpecs = unionLayerSpecLists(LayerSpecLists)
            % Returns a flat list of the uniquely-named LayerSpecs.
            LayerSpecs = vertcat(LayerSpecLists{:});
            Names = cellfun(@(Spec)Spec.Name, LayerSpecs, 'UniformOutput', false);
            [~, uniqueIndices] = unique(Names, 'stable');
            LayerSpecs = LayerSpecs(uniqueIndices);
end

function LayerSpecs = applyNameTableToLayers(NameTable, LayerSpecs)
    % For each LayerSpec, apply the NameTable to its input tensors.
    for L = 1:numel(LayerSpecs)
        for i = 1:numel(LayerSpecs{L}.InConns)
            LayerSpecs{L}.InConns{i} = nnet.internal.cnn.keras.util.renameConn(LayerSpecs{L}.InConns{i}, NameTable);
        end
    end
end

