classdef ClipLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable ...
    & nnet.internal.cnn.layer.CPUFusableLayer
    %ClipLayer
    %
    %   layer = ClipLayer(Name, Min, Max) creates a layer with name
    %   Name that clips the input between lower and upper bounds defined by
    %   Min and Max.
    
    %   Copyright 2018-2023 The MathWorks, Inc.
    properties
        Min
        Max
    end
    
    methods
        function this = ClipLayer(name, Min, Max)
            % TODO: get rid of these asserts 
            assert(isstring(name) || ischar(name));
            assert(~isempty(Min) && isnumeric(Min) && isreal(Min) && isscalar(Min))
            assert(~isempty(Max) && isnumeric(Max) && isreal(Max) && isscalar(Max))
            this.Name = name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:ClipDescription', num2str(Min), num2str(Max)));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:ClipType'));
            this.Min = Min;
            this.Max = Max;
        end
        
        function Z = predict(this, X)
            Z = min(this.Max, max(this.Min, X));
        end
        
        function dLdX = backward( this, X, Z, dLdZ, memory )
            dLdX = dLdZ;
            dLdX(X~=Z) = 0;
        end
    end

    methods (Static = true, Access = public, Hidden = true)       
         function name = matlabCodegenRedirect(~)
             name = 'nnet.internal.cnn.coder.keras.layer.ClipLayer';
         end
     end

    methods(Hidden)
        function layerArgs = getFusedArguments(layer)
            %getFusedArguments  Return arguments needed to call the
            % layer in a fused network
            layerArgs = { 'clip', layer.Min, layer.Max };
        end

        function tf = isFusable(~)
            %isFusable Flag if layer is fusable
            tf = true;
        end
    end
end