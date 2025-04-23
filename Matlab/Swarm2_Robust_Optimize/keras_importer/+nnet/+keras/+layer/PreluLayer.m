classdef PreluLayer < nnet.layer.Layer & nnet.internal.cnn.layer.CPUFusableLayer
    
    %   Copyright 2019-2023 The MathWorks, Inc.
    properties (Learnable)
        Alpha          % Alpha can be a scalar, or [1 ... 1 C]
    end
    
    properties(Hidden, SetAccess=protected)
        ChannelDim
        NumChannels
    end
    
    properties
        RawAlpha = [];
    end
        
    methods
        function this = PreluLayer(name, initialAlpha)
            % layer = PreluLayer(name, initialAlpha). 
            % initialAlpha can be a scalar or a vector. 
            
            % Changing alpha will result in re-computing the layer's 
            % internal state such as ChannelDim and NumChannels. 
            if ~(isstring(name) || ischar(name))
                error(message('nnet_cnn_kerasimporter:keras_importer:BadArgType2'));
            end
            
            this.Name        = name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:PreluLayerDescription'));
            this.Type        = getString(message('nnet_cnn_kerasimporter:keras_importer:PreluLayerType'));
            this.Alpha       = initialAlpha;
        end
        
        function Z = predict(layer, X)
            try 
                Z = max(0,X) + layer.Alpha.*min(0,X);
            catch ME 
                id = ME.identifier;
                if isequal(id, 'MATLAB:dimagree') || isequal(id, 'MATLAB:sizeDimensionsMustMatch')
                    throw(MException('nnet_cnn_kerasimporter:keras_importer:BadPReLUCustomAlphaShape',...
                        message('nnet_cnn_kerasimporter:keras_importer:BadPReLUCustomAlphaShape'))); 
                else 
                    rethrow(ME); 
                end
            end 
        end
        
        function [dLdX, dLdAlpha] = backward(layer, X, ~, dLdZ, ~)
            dLdX        = layer.Alpha .* dLdZ;
            posIdx      = X>0;
            dLdX(posIdx)= dLdZ(posIdx);
            nDims       = max(ndims(dLdZ), layer.ChannelDim);
            dimsToSum   = [1:layer.ChannelDim-1, layer.ChannelDim+1:nDims];
            dLdAlpha    = sum(min(0,X).*dLdZ, dimsToSum);
        end
        
        function this = set.Alpha(this, value) 
            [this, newAlpha] = this.computeChannelDimFromAlpha(value);
            this.Alpha = newAlpha;
        end
    end
    
    methods (Access = private)
        function [layer, newAlpha] = computeChannelDimFromAlpha(layer, alpha)
            % This function is called whenever alpha is set. The
            % internal state of the layer will need to change based on
            % the new value of alpha. 
            
            % Note: We are allowing the user to set the alpha to a
            % vector although the translator does not support this right
            % now.
            if isscalar(alpha)
                % all input dimensions share a single learnable alpha
                layer.NumChannels = 1;
                layer.ChannelDim = 0;
                newAlpha = alpha; 
            elseif isscalar(find(size(alpha) ~= 1))
                % This is reached when alpha is a vector along some dimension.
                layer.NumChannels = numel(alpha);
                layer.ChannelDim  = find(size(alpha) ~= 1); 
                newAlpha = alpha; 
            else
                % Matrix and higher order alpha is not supported. 
                error(message('nnet_cnn_kerasimporter:keras_importer:BadPReLUCustomAlphaShape')); 
            end
        end
    end

    methods (Static = true, Access = public, Hidden = true)       
         function name = matlabCodegenRedirect(~)
             name = 'nnet.internal.cnn.coder.keras.layer.PreluLayer';
         end
     end 

    methods(Hidden)
        function layerArgs = getFusedArguments(this)
            %getFusedArguments  Return arguments needed to call the
            % layer in a fused network
            layerArgs = { 'prelu', this.Alpha };
        end

        function tf = isFusable(~)
            %isFusable Flag if layer is fusable
            tf = true;
        end
    end
end
