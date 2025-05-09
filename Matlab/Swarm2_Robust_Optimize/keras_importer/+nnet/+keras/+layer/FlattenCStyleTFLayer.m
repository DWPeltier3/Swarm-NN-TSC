classdef FlattenCStyleTFLayer < nnet.layer.Layer & nnet.layer.Formattable
% FlattenCStyleTFLayer   FlattenCStyle TensorFlow layer
%
% layer = FlattenCStyleTFLayer(Name) creates a layer with name Name that
% flattens the input image into a vector, assuming C-style (or row-major)
% storage ordering of the input layer. Always returns a 2D output labelled CB.
% 
%   Copyright 2022-2023 The MathWorks, Inc.
%#codegen
    methods
        function this = FlattenCStyleTFLayer(name)
            this.Name = name;
            if coder.target('MATLAB')
                this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenDescription'));  
                this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenType'));                      
            end
        end
        
        function Z = predict( ~, X )
            fmt = dims(X);
            if coder.const(isequal(fmt,'SSSC') || isequal(fmt,'SSSCB'))
                % X is size [H W C D N].
                % Z is size [1 1 1 HWCD N].
                [sz1, sz2, sz3, sz4, sz5] = size(X);
                Z = reshape(permute(stripdims(X),[4 3 2 1 5]), [sz1*sz2*sz3*sz4 sz5]);
            elseif coder.const(isequal(fmt,'SC') || isequal(fmt,'SCB'))
                [sz1, sz2, sz3] = size(X); 
                Z = reshape(permute(stripdims(X), [2 1 3]), [sz1*sz2 sz3] ); 
            elseif coder.const(isequal(fmt,'CBT'))
                %Flatten with Sequence Input Layer
                %Common use case for 1D Layers
                %X: [C B T] ; Z:[CT B]
                [sz1, sz2, sz3] = size(X);
                Z = reshape(permute(stripdims(X),[1 3 2]), [sz1*sz3 sz2]);
            elseif coder.const(isequal(fmt,'SCBT'))
                %Flatten with Spatio-temporal data
                %X: [S C B T] ; Z:[SCT B]
                [sz1, sz2, sz3, sz4] = size(X);
                Z = reshape(permute(stripdims(X),[2 1 4 3]), [sz1*sz2*sz4 sz3]);
            elseif coder.const(isequal(fmt,'SSCBT'))
                %Flatten with Spatio-temporal data
                %X: [H W C B T] ; Z:[HWCT B]
                [sz1, sz2, sz3, sz4, sz5] = size(X);
                Z = reshape(permute(stripdims(X),[3 2 1 5 4]), [sz1*sz2*sz3*sz5 sz4]);
            elseif coder.const(isequal(fmt,'SSSCBT'))
                %Flatten with Spatio-temporal data
                %X: [H W D C B T] ; Z:[HWDCT B]
                [sz1, sz2, sz3, sz4, sz5, sz6] = size(X);
                Z = reshape(permute(stripdims(X),[4 3 2 1 6 5]), [sz1*sz2*sz3*sz4*sz6 sz5]);
            elseif coder.const(isequal(fmt,'SSC') || isequal(fmt,'SSCB'))
                % X is size [H W C N].
                % Z is size [1 1 HWC N].
                [sz1, sz2, sz3, sz4] = size(X);
                Z = reshape(permute(stripdims(X),[3 2 1 4]), [sz1*sz2*sz3 sz4]);                
            elseif coder.const(all(fmt(1:end-1) == 'S') && strcmp(fmt(end),'B'))
                % X is coming from an autogenerated custom layer: S*B
                % Codegen note:
                %   As codegen is not supported for autogenerated custom 
                %   layers, this branch will never be reached during codegen
                S = size(X);
                lastSDimIdx = numel(fmt)-1;
                Z = reshape(permute(stripdims(X),[lastSDimIdx:-1:1 numel(fmt)]), [prod(S(1:end-1)) S(end)]);
            elseif coder.const(isequal(fmt,'CB'))
                Z = X;
            else
                % Error if X has unexpected data format
                % Codegen note:
                %   As error occurs in MATLAB, this branch will never
                %   be reached during codegen
                throw(MException('nnet_cnn_kerasimporter:keras_importer:FlattenCStyleUnexpectedInput', ...
                    message('nnet_cnn_kerasimporter:keras_importer:FlattenCStyleUnexpectedInput', 'FlattenCStyleTFLayer')));
            end
            % Always apply CB labels.
            Z = dlarray(Z, "CB");            
        end
    end
end

