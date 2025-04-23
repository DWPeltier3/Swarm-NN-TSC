classdef PreluLayer < nnet.layer.Layer
    % Internal, codegen version of nnet.keras.layer.PreluLayer
    %#codegen
    
    %   Copyright 2021 The MathWorks, Inc.
    properties (Learnable)
        Alpha          % Alpha can be a scalar, or [1 ... 1 C]
    end
        
    methods
        function this = PreluLayer(name, Alpha)
            % layer = PreluLayer(name, Alpha). 
            % Alpha can be a scalar or a vector.
            this.Name        = name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:PreluLayerDescription'), matlab.internal.i18n.locale('en_US'));
            this.Type        = getString(message('nnet_cnn_kerasimporter:keras_importer:PreluLayerType'), matlab.internal.i18n.locale('en_US'));
            this.Alpha       = Alpha;
        end
        
        function Z = predict(layer, X)
                Z = max(0,X) + layer.Alpha.*min(0,X);             
        end        
    end

    methods (Static)
        function this_cg = matlabCodegenToRedirected(this)
            % Enable objects of class PreluLayer to be passed from MATLAB 
            % to the generated MEX function
            this_cg = nnet.internal.cnn.coder.keras.layer.PreluLayer(this.Name, this.Alpha);
        end
         
        function this = matlabCodegenFromRedirected(this_cg)
            % Enable objects of class PreluLayer to be successfully 
            % returned from the generated MEX function back to MATLAB
            this = nnet.keras.layer.PreluLayer(this_cg.Name, this_cg.Alpha);
        end
     end
end
