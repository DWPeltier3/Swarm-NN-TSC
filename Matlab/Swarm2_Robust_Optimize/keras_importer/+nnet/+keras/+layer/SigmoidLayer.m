classdef SigmoidLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable ...
    & nnet.internal.cnn.layer.CPUFusableLayer ...
    & nnet.internal.cnn.layer.BackwardOptional ...
    & nnet.layer.Elementwise

% SigmoidLayer   Sigmoid layer
%
%   layer = SigmoidLayer(Name) creates a sigmoid unit layer with
%   name Name. This type of layer calculates Y = 1./(1+exp(-X)).

%   Copyright 2017-2023 The MathWorks, Inc.
    methods
        function this = SigmoidLayer(Name)
            this.Name = Name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:SigmoidDescription'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:SigmoidType'));
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result
            if isdlarray(X)
                % Use the dlarray method when possible.  This will record a sigmoid
                % operation which will map back to this class when needed.
                Z = sigmoid(X);
            else
                Z = 1./(1+exp(-X));
            end
        end

        function dLdX = backward(~, ~, Z, dLdZ, ~)
            dLdX = dLdZ.*Z.*(1-Z);
        end

    end

    methods (Hidden)
        function layerArgs = getFusedArguments(~)
            layerArgs = { 'sigmoid' };
        end

        function tf = isFusable(~)
            % isFusable  Indicates if the layer is fusable in a given network.
            tf = true;
        end
    end
end
