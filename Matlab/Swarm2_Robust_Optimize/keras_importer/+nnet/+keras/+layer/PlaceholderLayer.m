classdef PlaceholderLayer < nnet.cnn.layer.PlaceholderLayer
%PlaceholderLayer   A layer to take the place of an unsupported Keras layer
% 
%   PlaceholderLayer properties:
%      KerasConfiguration	- A struct containing the Keras configuration for this layer.
%      Weights              - A struct containing weights imported from the Keras layer, if any.
% 
%   See also importKerasLayers.
% 
%   Copyright 2017-2022 The MathWorks, Inc.    
    
    properties
        KerasConfiguration     % A struct containing the Keras configuration for this layer.
        Weights = [];
        InputLabels = {};
        OutputLabels = {};
    end
    
    methods
        function this = PlaceholderLayer(name, KerasLayerType, KerasConfig, numInputs, numOutputs)
            assert(numInputs > 0);
            assert(numOutputs > 0);
            this@nnet.cnn.layer.PlaceholderLayer(name);
            this.KerasConfiguration = KerasConfig;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:PlaceholderDescription', KerasLayerType));
            this.Type = KerasLayerType;
            this.NumInputs = numInputs;
            this.NumOutputs = numOutputs;
            for i = 1:numInputs
                this.InputLabels{end+1} = '';
            end
            for i = 1:numOutputs
                this.OutputLabels{end+1} = '';
            end
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
