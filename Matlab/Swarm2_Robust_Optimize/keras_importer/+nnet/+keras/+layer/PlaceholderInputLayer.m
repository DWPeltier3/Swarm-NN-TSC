classdef PlaceholderInputLayer < nnet.cnn.layer.PlaceholderLayer    
%PlaceholderInputLayer   A layer to take the place of an unsupported Keras
%Input configuration. 
% 
%   PlaceholderInputLayer properties:
%      batch_input_shape	- An array containing the shape of the input as described by Keras.
%      InputRank            - Rank of the Input
%      KerasConfiguration	- A struct containing the Keras configuration for this layer.
%      OutputLabels         - DLT Labels for the output of the Placeholder layer
% 
%   See also importKerasLayers.
% 

% Copyright 2020-2022 The MathWorks, Inc.

    properties 
        batch_input_shape;
        InputRank;
        InputLabels = {''};
        OutputLabels = {''};
        KerasConfiguration;
    end 
    methods
        function this = PlaceholderInputLayer(name, batch_input_shape, KerasConfig)
            
            this@nnet.cnn.layer.PlaceholderLayer(name);
            if nargin == 3
                this.KerasConfiguration = KerasConfig;
            end
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:PlaceholderType'));
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:PlaceholderDescription', 'Input'));
            this.batch_input_shape = batch_input_shape;
            this.InputRank = numel(batch_input_shape);
        end

        function varargout = predict( ~, varargin )
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:PlaceholderCantBeUsed')));
        end
        
        function varargout = forward( ~, varargin )
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:PlaceholderCantBeUsed')));
        end
        
        function varargout = backward( ~, varargin )
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:PlaceholderCantBeUsed')));
        end

    end
end
