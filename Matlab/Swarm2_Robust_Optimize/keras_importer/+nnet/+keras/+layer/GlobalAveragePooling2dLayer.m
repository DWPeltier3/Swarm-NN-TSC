classdef GlobalAveragePooling2dLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable
    % GlobalAveragePooling2dLayer   GlobalAveragePooling2d layer
    %
    %   layer = GlobalAveragePooling2dLayer(Name) creates a layer
    %   with name Name that performs global average pooling. For an input
    %   layer of size [Height, Width, Channels], the layer output has size
    %   [1 1 Channels], and each output channel contains the average of
    %   that input channel across the entire image.
    
    %   Copyright 2017-2020 The MathWorks, Inc.
    methods
        function this = GlobalAveragePooling2dLayer(name)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:GlobalAveragePooling2dDescription'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:GlobalAveragePooling2dType'));
        end
        
        function Z = predict( this, X )
            % X is size [H W C N]. Z is size [1 1 C N].
            Z = mean(mean(X,1),2);
        end
        
        function dLdX = backward( this, X, Z, dLdZ, memory )
            % dLdZ is size [1 1 C N]. dLdX and X are size [H W C N]. 
            H = size(X,1);
            W = size(X,2);
            dLdX = repmat(dLdZ/(H*W), H, W); 
        end
    end    
end
