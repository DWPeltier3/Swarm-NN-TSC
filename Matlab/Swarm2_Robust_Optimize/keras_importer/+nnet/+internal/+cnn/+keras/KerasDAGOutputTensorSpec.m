classdef KerasDAGOutputTensorSpec < handle
    % Represents an output tensor. Keras represents a tensor as the {layer,
    % outputNum} pair that created that tensor.

% Copyright 2017 The Mathworks, Inc.

    properties
        OutputNum          % Which of the DAG's outputs is this? (origin 1)
        LayerName          % The unique name of the layer providing this output.
        LayerReplicaNum    % Which shared replica of the layer is creating the output?
        % 0 if it's a base layer, otherwise origin 1.
        LayerOutputNum     % Which output of the layer is creating the output?  (origin 1)
    end
    
    methods
        function this = KerasDAGOutputTensorSpec(OutputNum, Cell)
            % Cell =
            %   3×1 cell array
            %     {'leaky_re_lu_1'}
            %     {[            0]}
            %     {[            0]}
            this.OutputNum = OutputNum;
            this.LayerName = Cell{1};
            this.LayerReplicaNum = Cell{2};
            this.LayerOutputNum = Cell{3} + 1;
        end
    end
end
