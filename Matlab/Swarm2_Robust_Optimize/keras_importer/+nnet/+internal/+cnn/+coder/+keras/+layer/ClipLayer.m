classdef ClipLayer < nnet.layer.Layer 
    % Internal, codegen version of nnet.keras.layer.ClipLayer
    %#codegen
    
    %   Copyright 2021 The MathWorks, Inc.
    properties
        Min
        Max
    end
    
    methods
        function this = ClipLayer(name, Min, Max)
            this.Name = name;
            this.Min = Min;
            this.Max = Max;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:ClipDescription', num2str(Min), num2str(Max)), matlab.internal.i18n.locale('en_US'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:ClipType'), matlab.internal.i18n.locale('en_US'));
        end
        
        function Z = predict(this, X)
            Z = min(this.Max, max(this.Min, X));
        end        
    end

    methods (Static)
        function this_cg = matlabCodegenToRedirected(this)
            % Enable objects of class ClipLayer to be passed from MATLAB 
            % to the generated MEX function
            this_cg = nnet.internal.cnn.coder.keras.layer.ClipLayer(this.Name, this.Min, this.Max);
        end
         
        function this = matlabCodegenFromRedirected(this_cg)
            % Enable objects of class ClipLayer to be successfully 
            % returned from the generated MEX function back to MATLAB
            this = nnet.keras.layer.ClipLayer(this_cg.Name, this_cg.Min, this_cg.Max);
        end
     end
end
