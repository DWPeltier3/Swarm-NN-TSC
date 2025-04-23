function tf = isUnsupportedLayer(Layer)
    % Copyright 2021 The MathWorks, Inc.
    tf = isa(Layer, 'nnet.keras.layer.PlaceholderLayer');
end