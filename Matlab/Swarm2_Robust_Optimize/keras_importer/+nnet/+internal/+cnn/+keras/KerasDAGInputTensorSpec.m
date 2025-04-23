classdef KerasDAGInputTensorSpec < handle
    % Represents an input tensor. Keras represents a tensor as the {layer,
    % outputNum} pair that created that tensor.

% Copyright 2017 The Mathworks, Inc.

    properties
        InputNum         	% Which of the DAG's inputs is this? (origin 1)
        LayerName           % The unique name of the layer that creates this input.
        LayerReplicaNum     % Which shared replica of the layer is creating the input?
        % 0 if it's a base layer, otherwise origin 1.
        LayerOutputNum      % Which output of the layer is providing the input?  (origin 1)
    end
    
    methods
        function this = KerasDAGInputTensorSpec(InputNum, Cell)
            % Cell =
            %   4×1 cell array
            %     {'input_2' }
            %     {[       0]}
            %     {[       0]}
            %     {1×1 struct}
            %
            % Cell{4} =
            %   struct with no fields.
            this.InputNum = InputNum;
            this.LayerName = Cell{1};
            this.LayerReplicaNum = Cell{2};
            this.LayerOutputNum = Cell{3} + 1;
        end
    end
end
