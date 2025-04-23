classdef KerasOutputTensorSpec < handle

% Copyright 2019 The Mathworks, Inc.

    properties
        OutputNum          % Which of the DAG's outputs is this? (origin 1)
        LayerName          % The unique name of the layer providing this output.
        LayerReplicaNum    % Which shared replica of the layer is creating the output?
        % 0 if it's a base layer, otherwise origin 1.
        LayerOutputNum     % Which output of the layer is creating the output?  (origin 1)
    end
    
    methods
        function this = KerasOutputTensorSpec(OutputNum, Cell)
            % Cell =
            %   3◊1 cell array
            %     {'leaky_re_lu_1'}
            %     {0x0 double}
            %     {[            0]}
            this.OutputNum = OutputNum;
            this.LayerName = Cell{1};
            this.LayerReplicaNum = Cell{2};
            this.LayerOutputNum = Cell{3} + 1;
        end
    end
end