classdef KerasInputTensorSpec < handle

% Copyright 2019 The Mathworks, Inc.

    properties
        InputNum         	% Which of the Sequential's inputs is this? (always 1)
        LayerName           % The unique name of the layer that creates this input.
        LayerReplicaNum     % Which shared replica of the layer is creating the input?
        % 0 if it's a base layer, otherwise origin 1.
        LayerOutputNum      % Which output of the layer is providing the input?  (always 1)
    end
    
    methods
        function this = KerasInputTensorSpec(InputNum, Cell)
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
            if iscell(Cell)
                this.LayerName = Cell{1};
                this.LayerReplicaNum = Cell{2};
                this.LayerOutputNum = Cell{3} + 1;
            else
                this.LayerName = Cell;
            end            
        end
    end
end