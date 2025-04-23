classdef SoftsignLayer < nnet.layer.Layer...
                        & nnet.internal.cnn.layer.Traceable ...
                        & nnet.internal.cnn.layer.CPUFusableLayer ...
                        & nnet.layer.Acceleratable ...
                        & nnet.layer.Elementwise

% SoftsignLayer   Softsign layer
%
%   layer = SoftsignLayer(Name) creates a softsign activation layer with
%   name Name. This type of layer calculates Y = X ./ (1 + abs(X)).

%   Copyright 2023 The MathWorks, Inc.
    methods
        function this = SoftsignLayer(Name)
            this.Name = Name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:SoftsignDescription'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:SoftsignType'));
        end
        
        function Z = predict(~, X)
            Z = X ./ (1 + abs(X));
        end        
    end

    methods (Hidden)
        function layerArgs = getFusedArguments(~)
            % getFusedArguments  Returned the arguments needed to call the
            % layer in a fused network.
            layerArgs = { 'softsign' };
        end

        function tf = isFusable(~)
            % isFusable  Indicates if the layer is fusable in a given network.
            tf = true;
        end
    end

    methods (Static = true, Access = public, Hidden = true)       
         function name = matlabCodegenRedirect(~)
             name = 'nnet.internal.cnn.coder.keras.layer.SoftsignLayer';
         end
    end 
end
