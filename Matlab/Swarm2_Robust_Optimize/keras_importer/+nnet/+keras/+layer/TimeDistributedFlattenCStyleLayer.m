classdef TimeDistributedFlattenCStyleLayer < nnet.layer.Layer & nnet.layer.Formattable
% TimeDistributedFlattenCStyleLayer   
%
% layer = TimeDistributedFlattenLayer(Name) creates a layer with name Name that
% flattens the sequence of input image into a sequence of vector, assuming C-style (or row-major)
% storage ordering of the input layer.
% 
%
% For a sequence of images folded coming in, the input shape is [H, W, C,
% NS]. In this case, the output shape should be [HWC, NS] so that the
% output can be unfolded to [HWC, N, S] in sequence unfolding layer.
    
    %   Copyright 2019-2023 The MathWorks, Inc.
    methods
        function this = TimeDistributedFlattenCStyleLayer(name)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenDescription'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenType'));
        end
        
        function Z = predict( ~, X )
            if ismember(dims(X), ["SSC", "SSCB"])
                % When input is a batch of images
                % X is size [H W C NS].
                % Z is size [HWC NS].
                [sz1, sz2, sz3, sz4] = size(X);
                Z = reshape(permute(stripdims(X),[3 2 1 4]), [sz1*sz2*sz3 sz4]);
            elseif isequal(dims(X), 'CB')
                % When input is a batch of vectors, serve as linear
                Z = X;
            else
                % Error if X has unexpected data format
                throw(MException('nnet_cnn_kerasimporter:keras_importer:FlattenCStyleUnexpectedInput', ...
                    message('nnet_cnn_kerasimporter:keras_importer:FlattenCStyleUnexpectedInput', 'TimeDistributedFlattenCStyleLayer')));
            end
            % Always apply CB labels.
            Z = dlarray(Z, 'CB');
        end
    end

    methods (Static = true, Access = public, Hidden = true)       
         function name = matlabCodegenRedirect(~)
             name = 'nnet.internal.cnn.coder.keras.layer.TimeDistributedFlattenCStyleLayer';
         end
     end 
end

