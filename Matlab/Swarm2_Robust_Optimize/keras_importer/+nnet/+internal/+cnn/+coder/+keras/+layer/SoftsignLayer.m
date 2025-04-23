classdef SoftsignLayer < nnet.layer.Layer
 % Internal, codegen version of nnet.keras.layer.SoftsignLayer
 %#codegen

%   Copyright 2023 The MathWorks, Inc.
    methods
        function this = SoftsignLayer(Name)
            this.Name = Name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:SoftsignDescription'), matlab.internal.i18n.locale('en_US'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:SoftsignType'), matlab.internal.i18n.locale('en_US'));
        end
        
        function X = predict(~, X)
            parfor i = 1:numel(X)
                X(i) = X(i) ./ (1 + abs(X(i)));
            end
        end        
    end

    methods (Static)
        function this_cg = matlabCodegenToRedirected(this)
            % Enable objects of class SoftsignLayer to be passed from MATLAB 
            % to the generated MEX function
            this_cg = nnet.internal.cnn.coder.keras.layer.SoftsignLayer(this.Name);
        end
         
        function this = matlabCodegenFromRedirected(this_cg)
            % Enable objects of class SoftsignLayer to be successfully 
            % returned from the generated MEX function back to MATLAB
            this = nnet.keras.layer.SoftsignLayer(this_cg.Name);
        end
     end
end
