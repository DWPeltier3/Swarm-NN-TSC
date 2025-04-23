classdef TanhLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable ...
    & nnet.internal.cnn.layer.CPUFusableLayer ...
    & nnet.internal.cnn.layer.BackwardOptional ...
    & nnet.layer.Elementwise

% TanhLayer   Tanh layer
%
%   layer = TanhLayer(Name) creates a tanh unit layer with
%   name Name. This type of layer calculates Y = tanh(X).

%   Copyright 2017-2023 The MathWorks, Inc.
    methods
        function this = TanhLayer(Name)
            this.Name = Name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:TanhDescription'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:TanhDescription'));
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result
            Z = tanh(X);
        end

        function dLdX = backward(~, ~, Z, dLdZ, ~)
            % Backward propagate the derivative of the loss function through
            % the layer
            dLdX = (1 - Z.^2) .* dLdZ;
        end 
    end

    methods (Hidden)
        function layerArgs = getFusedArguments(~)
            % getFusedArguments  Returned the arguments needed to call the
            % layer in a fused network.
            layerArgs = { 'tanh' };
        end

        function tf = isFusable(~)
            % isFusable  Indicates if the layer is fusable in a given network.
            tf = true;
        end
    end
end

