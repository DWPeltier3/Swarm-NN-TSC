function [layers, name] = getLayersFromSequentialModel(SequentialModelStructConfig)
    
% Copyright 2019 The Mathworks, Inc.
    

%% Keras version < 2.2.3:
% Struct =
%   struct with fields:
%         config: [18×1 struct]
%     class_name: 'Sequential'

%% Keras version >= 2.2.3:
% Struct =
%   struct with fields:
%         config: [1×1 struct]
%     class_name: 'Sequential'
%
% Struct.config =
%   struct with fields:
%       name: 'sequential_1'
%     layers: [18×1 struct]
%%

if iUsingKeras223OrLater(SequentialModelStructConfig)
    layers = SequentialModelStructConfig.layers;
    name = SequentialModelStructConfig.name;
else
    layers = SequentialModelStructConfig;
    name = '';
end
end

function tf = iUsingKeras223OrLater(SequentialModelStructConfig)
tf = isfield(SequentialModelStructConfig, 'layers');
end