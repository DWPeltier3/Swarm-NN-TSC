classdef ConverterForCrop2DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSC[B][T]
    methods
        function convertedLayer = toTensorflow(this)
            inputImageSize = this.InputSize{1};              % HWC
            referenceImageSize = this.InputSize{2};          % HWC
            if isequal(this.Layer.Mode, 'centercrop')
                % Convert location to numeric upper left front corner
                sz           = inputImageSize(1:2);
                outputSize	 = referenceImageSize(1:2);
                % Compare the following to nnet.internal.cnn.layer.util.Crop2DCenterCropStrategy
                centerX      = floor(sz(1:2)/2 + 1);
                centerWindow = floor(outputSize/2 + 1);
                HWLocation     = centerX - centerWindow + 1;
            else
                % Location is [x,y], which is [W,H]
                HWLocation = this.Layer.Location([2,1]);
            end
            topcrop = HWLocation(1) - 1;
            bottomcrop = inputImageSize(1) - referenceImageSize(1) - topcrop;
            leftcrop = HWLocation(2) - 1;
            rightcrop = inputImageSize(2) - referenceImageSize(2) - leftcrop;

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName(1), this.OutputTensorName,...
                "layers.Cropping2D", "cropping=((%d,%d),(%d,%d))", {topcrop, bottomcrop, leftcrop, rightcrop},...
                "", this.layerAnalyzer.IsTemporal);

        end
    end
end
