classdef ConverterForTransposedConvolution1DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % According to the DLT Doc, this layer can handle these input formats:
    % in DLT:                   in Keras:
    % CBT (conv over T)         BTC
    % SCB (conv over S)         BSC
    % SCBT (conv over S)        BTSC
    %
    % But it can also handle these dlnetwork-only formats:
    %
    % CT                        BTC
    % SC                        BSC
    % SCT                       BTSC

    % According to the TF doc, the input is always interpreted as:
    %       3D tensor with shape: (batch_size, steps, channels)
    %
    % 3D tensor with shape: (batch_size, new_steps, filters) If output_padding is specified:
    %     new_timesteps = ((timesteps - 1) * strides + kernel_size -
    %     2 * padding + output_padding)
    % 
    % It only accepts and returns 3D tensors. So to implement BTSC I/O we
    % wrap it in TimeDistributed. It always convs over the second dim,
    % which conveniently coincides with MATLAB in the BTC and BSC cases.
    methods
        function convertedLayer = toTensorflow(this)
            filters = this.Layer.NumFilters;
            kernel_size = this.Layer.FilterSize; 
            if this.Layer.Stride==1
                strideStr = '';
            else
                strideStr = sprintf(", strides=%s", string(this.Layer.Stride));
            end
            if isequal(this.Layer.CroppingMode, 'same')
                paddingStr = ', padding="same"';
            else
                % padding="valid". omit
                paddingStr = '';                                                % A padding layer will be added below.
            end
            UseTimeDistributed = ismember(this.InputFormat, ["SCBT", "SCT"]);

            % Generate code
            layerName = this.OutputTensorName + "_";
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if isequal(this.Layer.CroppingMode, 'same') || all(this.Layer.CroppingSize(:) == 0)
                % Generate only a conv line
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.Conv1DTranspose", "%d, %d%s%s", {filters, kernel_size, strideStr, paddingStr},...
                    layerName, UseTimeDistributed);
            else
                % Generate a conv line and a cropping line. 1D cropping
                % layout is [l r]
                l = this.Layer.CroppingSize(1);
                r = this.Layer.CroppingSize(2);
                convertedLayer.layerCode = [
                    kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.Conv1DTranspose", "%d, %d%s%s", {filters, kernel_size, strideStr, paddingStr},...
                    layerName, UseTimeDistributed)

                    kerasCodeLine(this, this.OutputTensorName, this.OutputTensorName,...
                    "layers.Cropping1D", "cropping=(%d,%d)", {l, r},...
                    "", UseTimeDistributed)
                    ];
            end
            % Format of kernel weights:
            % Keras and MATLAB are both:  k x F x C
            kerasWeights = permute(this.Layer.Weights, [3 2 1]);   % Switch memory ordering from Col-major to row-major.
            convertedLayer.LayerName = layerName;
            convertedLayer.weightNames     = ["kernel", "bias"];
            convertedLayer.weightArrays    = {kerasWeights; this.Layer.Bias};                                                 % A matrix and a vector.
            convertedLayer.weightShapes    = {size(this.Layer.Weights, 1:3); numel(this.Layer.Bias)};  	% 3D and 1D.
        end
    end
end
