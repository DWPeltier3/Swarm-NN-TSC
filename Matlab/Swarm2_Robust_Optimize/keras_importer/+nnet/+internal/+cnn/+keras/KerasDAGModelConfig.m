classdef KerasDAGModelConfig < nnet.internal.cnn.keras.KerasModelConfig

    % Copyright 2017-2023 The Mathworks, Inc.
    methods
        function this = KerasDAGModelConfig(Struct, isTensorFlowModel)
            % Struct =
            %   struct with fields:
            %
            %            layers: [9×1 struct]
            %      input_layers: {{3×1 cell}}
            %     output_layers: {{3×1 cell}}
            %              name: 'model_4'
            this.Name = Struct.name;

            numLayers = ones(numel(Struct.layers), 1);
            this.Layers = arrayfun(@nnet.internal.cnn.keras.KerasLayerInsideDAGModel, Struct.layers, logical(numLayers .* isTensorFlowModel), 'UniformOutput', false);
            if isstruct(Struct.input_layers) 
                Struct.input_layers = struct2cell(Struct.input_layers); 
            end 
            this.InputLayers = cellfun(@nnet.internal.cnn.keras.KerasInputTensorSpec, num2cell(1:numel(Struct.input_layers))', Struct.input_layers, 'UniformOutput', false);
            if isstruct(Struct.output_layers)
                Struct.output_layers = struct2cell(Struct.output_layers); 
            end
            this.OutputLayers = cellfun(@nnet.internal.cnn.keras.KerasOutputTensorSpec, num2cell(1:numel(Struct.output_layers))', Struct.output_layers, 'UniformOutput', false);
        end
    end
end
