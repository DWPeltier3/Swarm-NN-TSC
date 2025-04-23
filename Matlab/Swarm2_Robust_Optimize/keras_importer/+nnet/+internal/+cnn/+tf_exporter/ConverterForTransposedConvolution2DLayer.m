classdef ConverterForTransposedConvolution2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSC[B][T], SC[B]T (spatio-temporal)
    methods
        function convertedLayer = toTensorflow(this)
            numFilters = this.Layer.NumFilters;
            kernel_size = sprintf("(%s)", join(string(this.Layer.FilterSize), ','));        % (H,W)
            if all(this.Layer.Stride==1)
                strideStr = '';
            else
                strideStr = sprintf(", strides=(%s)", join(string(this.Layer.Stride), ','));            % (H,W)
            end
            if isequal(this.Layer.CroppingMode, 'same')
                paddingStr = ', padding="same"';
            else
                % padding="valid". omit
                paddingStr = '';                                                % A padding layer will be added below.
            end

            % Generate code
            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if isequal(this.Layer.CroppingMode, 'same') || all(this.Layer.CroppingSize(:) == 0)
                % Generate only a conv line
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.Conv2DTranspose", "%d, %s%s%s", {numFilters, kernel_size, strideStr, paddingStr},...
                    layerName, this.layerAnalyzer.IsTemporal);
            else
                % Generate a conv line and a cropping line. 2D cropping
                % layout is [t b l r]
                t = this.Layer.CroppingSize(1);
                b = this.Layer.CroppingSize(2);
                l = this.Layer.CroppingSize(3);
                r = this.Layer.CroppingSize(4);
                convertedLayer.layerCode = [
                    kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.Conv2DTranspose", "%d, %s%s%s", {numFilters, kernel_size, strideStr, paddingStr},...
                    layerName, this.layerAnalyzer.IsTemporal)

                    kerasCodeLine(this, this.OutputTensorName, this.OutputTensorName,...
                    "layers.Cropping2D", "cropping=((%d,%d),(%d,%d))", {t, b, l, r},...
                    "", this.layerAnalyzer.IsTemporal)
                    ];
            end
            % Format of kernel weights:
            % Keras and MATLAB are both:  H x W x F x C
            kerasWeights = permute(this.Layer.Weights, [4 3 2 1]);   % Switch memory ordering from Col-major to row-major.
            convertedLayer.LayerName = layerName;
            convertedLayer.weightNames     = ["kernel", "bias"];
            convertedLayer.weightArrays    = {kerasWeights; this.Layer.Bias};                                                 % A matrix and a vector.
            convertedLayer.weightShapes    = {size(this.Layer.Weights, 1:4); numel(this.Layer.Bias)};  	% 4D and 1D.
        end
    end
end
