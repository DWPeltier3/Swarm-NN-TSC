function NNTName = makeNNTName(KerasLayerName)
% Make the Keras layer name compatible with NNT. 
% * Change forward slashes to vertical bars
% KerasLayerName is a char vector.
NNTName = strrep(KerasLayerName, '/', '|');
end