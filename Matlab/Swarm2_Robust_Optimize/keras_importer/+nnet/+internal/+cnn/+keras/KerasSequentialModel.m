classdef KerasSequentialModel < nnet.internal.cnn.keras.KerasModel
    
    % Copyright 2017-2023 The Mathworks, Inc.
    
    properties
        isDAG = false;
    end
    
    methods
        function this = KerasSequentialModel(SequentialModelStruct, isTensorFlowModel)
            if nargin == 2
                this.isTensorFlowModel = isTensorFlowModel;
            end
            this.ClassName  = SequentialModelStruct.class_name;
            this.Config = nnet.internal.cnn.keras.KerasSequentialModelConfig(SequentialModelStruct.config, isTensorFlowModel);
            
            this.isTimeDistributed = hasTimeDistributed(this.Config);
            this.isDAG = hasDAG(this.Config) || this.isTimeDistributed;
        end
        
        function [LayerSpecs, InputLayerIndices, OutputTensors] = flatten(this)
            % this.Config is a KerasSequentialModelConfig, flattenLayer is
            % in class of KerasModelConfig
            [LayerSpecs, NameTable] = flattenLayer(this.Config, '', this.isTensorFlowModel);
            % Create output tensors and apply the NameTable to them
            OutputTensors = {nnet.internal.cnn.keras.Tensor(LayerSpecs{end}.Name, 1)};
            OutputTensors = cellfun(@(Conn)nnet.internal.cnn.keras.util.renameConn(Conn, NameTable), OutputTensors, 'UniformOutput', false);
            InputLayerIndices = 1;
        end
    end
end

function tf = hasDAG(modelConfig)
    % layers is a cell array of KerasLayerInsideDAG/SequentialModel
    tf = false;
    if isa(modelConfig, 'nnet.internal.cnn.keras.KerasDAGModelConfig')
        tf = true;
    elseif isa(modelConfig, 'nnet.internal.cnn.keras.KerasSequentialModelConfig')
        for L = 1:numel(modelConfig.Layers)
            tf = tf || hasDAG(modelConfig.Layers{L}.Config);
            if tf == true
                break;
            end
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
