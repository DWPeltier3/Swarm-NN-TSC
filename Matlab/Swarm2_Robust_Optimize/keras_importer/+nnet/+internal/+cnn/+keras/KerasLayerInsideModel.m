classdef KerasLayerInsideModel < handle
   
    % Copyright 2019-2023 The Mathworks, Inc.
    
    properties
        % Raw keras objects
        Name
        ClassName
        % 'Config': 
        % If KerasClassName is a base layer, it is a KerasBaseLevelLayerConfig. 
        % If KerasClassName is Model, it is a KerasDAGModelConfig.
        % If KerasClassName is Sequential, it is a cell array of KerasSequentialModelConfig
        Config
        InboundConnections = {};    % Cell array of KerasDAGInputTensorSpec
        IsTimeDistributed = false;
    end
    
    methods
        function [LayerSpecs, NameTable] = expandLayer(this, SubmodelName, isTensorFlowModel)
        % Layer is a KerasLayerInsideModel
        % Only expand the TimeDistributed Wrapper for dlnetworks
        if any(strcmp(this.ClassName, {'Sequential', 'Model', 'TimeDistributed'})) &&  isprop(this.Config, 'IsTimeDistributed')
                [LayerSpecs, NameTable] = expandSubmodel(this, SubmodelName, isTensorFlowModel);
        else
            % It's a base layer. No expansion.
            LSpec = nnet.internal.cnn.keras.LayerSpec.fromBaseLayer(this, SubmodelName, isTensorFlowModel);
            LSpec.InConns = cellfun(@nnet.internal.cnn.keras.util.connFromTensorSpec, this.InboundConnections, 'UniformOutput', false);
            LayerSpecs = {LSpec};   
            NameTable = table;
        end
        end
    end
    
    methods(Abstract)
        % This function is used to specify how submodel connects with
        % inbounds when it is expanded.
        this = connectSubmodelInbound(this);
    end
    
    methods(Access=protected)
        function [LayerSpecs, NameTable] = expandSubmodel(this, SubmodelName, isTensorFlowModel)
            if isempty(SubmodelName)
                SubmodelName = this.Name;
            end
            this = connectSubmodelInbound(this);
            [LayerSpecs, RecursiveNametable] = flattenLayer(this.Config, SubmodelName, isTensorFlowModel);
            NameTable = nameTableFromLayer(this);
            NameTable = vertcat(NameTable, RecursiveNametable);
        end
        
        function NameTable = nameTableFromLayer(this)
            % Create a nametable entry for each element of layer.Config.OutputLayers.
            % layer is a KerasLayerInsideModel
            OutputLayerSpecs = this.Config.OutputLayers;
            NameTable = table;
            if isa(this.Config, 'nnet.internal.cnn.keras.KerasSequentialModelConfig') && isempty(this.Config.Name)
                name = this.Name;
            else
                name = this.Config.Name;
            end
            NameTable.FromName = repmat({name}, numel(OutputLayerSpecs), 1);
            NameTable.FromNum = cellfun(@(Spec)Spec.OutputNum, OutputLayerSpecs);
            NameTable.ToName = cellfun(@(Spec)Spec.LayerName, OutputLayerSpecs, 'UniformOutput', false);
            NameTable.ToNum = cellfun(@(Spec)Spec.LayerOutputNum, OutputLayerSpecs);
        end
    end
end
