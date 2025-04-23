classdef TimeDistributedFlattenCStyleLayer < nnet.layer.Layer & nnet.layer.Formattable
% Internal, codegen version of nnet.keras.layer.TimeDistributedFlattenCStyleLayer
%#codegen
    
    %   Copyright 2021-2023 The MathWorks, Inc.
    methods
        function this = TimeDistributedFlattenCStyleLayer(name)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenDescription'), matlab.internal.i18n.locale('en_US'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenType'), matlab.internal.i18n.locale('en_US'));
        end
        
        function Z = predict( ~, X )
            fmt = dims(X);
            if coder.const(isequal(fmt,'SSC') || isequal(fmt,'SSCB'))
                % When input is a batch of images
                % X is size [H W C NS].
                % Z is size [HWC NS].
                [sz1, sz2, sz3, sz4] = size(X);
                Z = reshape(permute(stripdims(X),[3 2 1 4]), [sz1*sz2*sz3 sz4]);
            elseif coder.const(isequal(dims(X), 'CB'))
                % When input is a batch of vectors, serve as linear
                Z = X;
            else
                % Error if X has unexpected data format
                % Codegen note:
                %   As an equivalent error occurs in MATLAB, this branch 
                %   will never be reached during codegen
                coder.internal.assert(false,'nnet_cnn_kerasimporter:keras_importer:FlattenCStyleUnexpectedInput','TimeDistributedFlattenCStyleLayer');               
            end
             % Always apply CB labels.
            Z = dlarray(Z, 'CB');
        end
    end

    methods (Static)
        function this_cg = matlabCodegenToRedirected(this)
            % Enable objects of class TimeDistributedFlattenCStyleLayer to be passed from MATLAB 
            % to the generated MEX function
            this_cg = nnet.internal.cnn.coder.keras.layer.TimeDistributedFlattenCStyleLayer(this.Name);
        end
         
        function this = matlabCodegenFromRedirected(this_cg)
            % Enable objects of class TimeDistributedFlattenCStyleLayer to be successfully 
            % returned from the generated MEX function back to MATLAB
            this = nnet.keras.layer.TimeDistributedFlattenCStyleLayer(this_cg.Name);
        end
     end
end

