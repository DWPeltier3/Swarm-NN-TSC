classdef ConvertedLayer

    %   Copyright 2022 The MathWorks, Inc.

    properties
        % Set LayerName if the layer has weights
        LayerName   string

        % Tensorflow code for the layer. Each string in this string array
        % ends up as one line of code in the model.py file.
        layerCode   string

        % Names of the weights for the layer. These can be any
        % python-compatible names.
        weightNames    string

        % A cell array of the python shapes of the layer weights, in the
        % same ordering as weightNames. Each shape is a numeric vector.
        weightShapes    cell

        % A cell array of weight arrays for the layer, in the same ordering
        % as weightNames. Each array must be in row-major storage ordering,
        % so that it can simply be reshaped in python into the
        % corresponding 'weightShapes' entry. TIP: Given an array W in matlab
        % with size [a,b,c], set its 'weightShape' to be [a,b,c] and its
        % 'weightArray' to be permute(W, [3 2 1]). This converts it from
        % column-major to row-major ordering.
        weightArrays    cell

        % An array of warning 'message' objects for all warnings that
        % occurred during conversion. The text of these messages will be
        % written into the README.txt file
        WarningMessages message

        % The classnames of any pre-written Tensorflow custom layers
        % referenced in the generated layer code. Setting this property
        % will cause the source code for those layers to be written into
        % the model.py file.
        customLayerNames    string

        % The names of any additional python packages referenced in the
        % generated layer code, e.g., "tf_addons"
        packagesNeeded  string

        % A flag indicating whether the converson into runnable lines of TF
        % code was successful. If false, the layer will be treated as
        % unsupported and have a custom layer template generated for it in
        % python. Set to false if your layer had settings for which
        % conversion is unsupported. For example, this is set by
        % ConverterForMaxPooling2DLayer when HasUnpoolingOuptuts=true.
        Success logical = true

        % If this layer had to rename a network input tensor, specify the
        % old and new names here as a string array ["oldname", "newname"].
        % NOTE: This is only used by Converters for input layers that have
        % normalizations.
        RenameNetworkInputTensor   string

        % Classname and code to implement an auto-generated placeholder
        % custom layer. NOTE: This is only to used by
        % ConverterForUnsupportedLayer.
        placeholderLayerName    string
        placeholderLayerCode    string
    end

    methods
        function checkLayer(this)
            % These error messages are for developers' eyes only.
            % Check weights:
            assert(isempty(this.weightArrays) || ~isempty(this.LayerName), ...
                "LayerName must be nonempty if the layer has weights");
            assert(~any(cellfun(@isempty, this.weightShapes)), ...
                "Do not use 0D python shapes for weights. Use the 1D shape of [1] instead.");
            assert(numel(this.weightNames) == numel(this.weightArrays) && numel(this.weightArrays) == numel(this.weightShapes),...
                "weightNames, weightArrays, and weightShapes must all be the same length");
            assert(all(cellfun(@isnumeric, this.weightArrays)),...
                "All weightArrays must be numeric");
            assert(all(cellfun(@(array, shape)numel(array)==prod(shape), this.weightArrays, this.weightShapes)),...
                "All weightArrays must have the number of elements that their Shape specifies");
            % Check other:
            assert(isempty(this.placeholderLayerName) == isempty(this.placeholderLayerCode),...
                "placeholderLayerName and placeholderLayerCode must either be both empty or both non-empty");
            assert(isempty(this.RenameNetworkInputTensor) || numel(this.RenameNetworkInputTensor)==2,...
                "RenameNetworkInputTensor must either be empty or a string array of length 2");
        end
    end
end