classdef KerasModelConfig < handle
    
    % Copyright 2019-2021 The Mathworks, Inc.
    
    properties
        Name
        Layers              % A cellarray of KerasLayerInsideModel
        InputLayers         % A cellarray of KerasInputTensorSpec
        OutputLayers        % A cellarray of KerasOutputTensorSpec
        IsTimeDistributed = false;
    end
    
    methods
        function [LayerSpecs, NameTable] = flattenLayer(this, SubmodelName, isTFModel)
            % Expand each layer of the Sequential into a flat list of LayerSpecs
            % L is KerasLayerInsideModel
            this.Layers = cellfun(@(L)setfield(L, 'IsTimeDistributed', this.IsTimeDistributed), this.Layers, 'UniformOutput', false);
            [ExpandedLayerSpecs, ExpandedNameTables] = cellfun(@(L)expandLayer(L, SubmodelName, isTFModel), ...
            this.Layers, 'UniformOutput', false);
            % Concatenate all the LayerSpec lists and NameTables
            LayerSpecs = unionLayerSpecLists(ExpandedLayerSpecs);
            % LayerSpecs = vertcat(ExpandedLayerSpecs{:});
            NameTable = vertcat(ExpandedNameTables{:});
            % Apply NameTable to the LayerSpecs
            LayerSpecs = applyNameTableToLayers(NameTable, LayerSpecs);
        end
        
        function visited = checkWeightSharing(this, visited)
            % If there's any layer that occurs more than once in the
            % layergraph, then it would be regarded as weighting sharing
            % and throw an error.
            for L = 1:numel(this.Layers)
                thisLayer = this.Layers{L};
                if ~isequal(thisLayer.ClassName, 'InputLayer')
                    name = thisLayer.Name;
                    if any(ismember(visited, name))
                        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:NoWeightSharing')));
                    else
                        visited{end+1} = name;
                    end
                end
                if ~isa(thisLayer.Config, 'nnet.internal.cnn.keras.KerasBaseLevelLayerConfig')
                   visited = checkWeightSharing(thisLayer.Config, visited);
                end
            end
        end
    end
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
