classdef KerasDAGModel < nnet.internal.cnn.keras.KerasModel

% Copyright 2017-2023 The Mathworks, Inc.

    methods
        function this = KerasDAGModel(Struct, isTensorFlowModel)
            % Struct =
            %   struct with fields:
            %     class_name: 'Model'
            %         config: [1 * 1 struct]
            if nargin == 2
                this.isTensorFlowModel = isTensorFlowModel;
            end
            this.ClassName = Struct.class_name;
            this.Config = nnet.internal.cnn.keras.KerasDAGModelConfig(Struct.config, isTensorFlowModel);
            this.isTimeDistributed = hasTimeDistributed(this.Config);
        end
        
        function [LayerSpecs, InputLayerIndices, OutputTensors] = flatten(this)
            % this.Config is a KerasDAGConfig, flattenLayer is
            % in class of KerasModelConfig
            [LayerSpecs, NameTable] = flattenLayer(this.Config, '', this.isTensorFlowModel);
            % Create output tensors and apply the NameTable to them
            OutputTensors = cellfun(@nnet.internal.cnn.keras.util.connFromTensorSpec, this.Config.OutputLayers, 'UniformOutput', false);
            OutputTensors = cellfun(@(Conn)nnet.internal.cnn.keras.util.renameConn(Conn, NameTable), OutputTensors, 'UniformOutput', false);
            % Find input layers
            AllLayerNames = cellfun(@(LS)LS.Name, LayerSpecs, 'UniformOutput', false);
            InputLayerNames = cellfun(@(TS)TS.LayerName, this.Config.InputLayers, 'UniformOutput', false);
            InputLayerIndices = find(ismember(AllLayerNames, InputLayerNames));
        end
    end
end

function tf = hasTimeDistributed(modelConfig)
    % layers is a cell array of KerasLayerInsideDAG/SequentialModel
    tf = false;
    if isa(modelConfig, 'nnet.internal.cnn.keras.KerasBaseLevelLayerConfig')
        tf = false;
    elseif ~isa(modelConfig, 'nnet.internal.cnn.keras.KerasTimeDistributedModelConfig')
        for L = 1:numel(modelConfig.Layers)
            tf = tf || hasTimeDistributed(modelConfig.Layers{L}.Config);
            if tf == true
                break;
            end
        end
    else
        tf = true;
    end
end
