classdef PlaceholderOutputLayer < nnet.cnn.layer.PlaceholderLayer
%PlaceholderOutputLayer   A layer to take the place of an unsupported Keras
%loss or output layer configuration. 
% 
%   See also importKerasLayers.
% 
properties
    InputLabels = {''};
    OutputLabels = {''};
end

% Copyright 2020-2023 The MathWorks, Inc.
    methods
        function this = PlaceholderOutputLayer(name)
            this@nnet.cnn.layer.PlaceholderLayer(name);
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:PlaceholderDescription', 'Loss'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:PlaceholderType'));
        end
    end
end
