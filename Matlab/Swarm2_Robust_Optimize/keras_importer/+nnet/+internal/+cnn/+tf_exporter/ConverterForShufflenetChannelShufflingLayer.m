classdef ConverterForShufflenetChannelShufflingLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.


    % Layer source code in matlab:
    %         function Z = predict(layer, X)
    %             % Shuffle
    %             [h, w, c, n] = size(X);
    %             g = layer.Groups; % Number of groups
    %             f = c/g; % Number of channels per group
    %             Z = reshape(X, [h, w, f, g, n]);
    %             Z = permute(Z, [1 2 4 3 5]);
    %             Z = reshape(Z, size(X));
    %         end

    methods
        function convertedLayer = toTensorflow(this)
            groups = this.Layer.Groups;

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = sprintf("%s = ChannelShufflingLayer(%d, name=""%s"")(%s)", this.OutputTensorName, groups, this.OutputTensorName, this.InputTensorName);
            % Include a custom layer
            convertedLayer.customLayerNames = "ChannelShufflingLayer";
        end
    end
end
