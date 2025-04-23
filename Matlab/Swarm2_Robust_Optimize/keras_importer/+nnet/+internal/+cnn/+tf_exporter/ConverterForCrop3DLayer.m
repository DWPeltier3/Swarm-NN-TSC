classdef ConverterForCrop3DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported input formats: SSSC[B][T]
    methods
        function convertedLayer = toTensorflow(this)
            inputImageSize = this.InputSize{1};              % HWDC
            referenceImageSize = this.InputSize{2};          % HWDC
            CropLocation = this.Layer.CropLocation;                                      % [Top, Left, front] of included region.
            if isequal(CropLocation, 'centercrop')
                % Convert location to numeric upper left front corner
                sz           = inputImageSize(1:3);
                outputSize	 = referenceImageSize(1:3);
                % Compare the following to nnet.internal.cnn.layer.util.Crop2DCenterCropStrategy
                centerX      = floor(sz(1:3)/2 + 1);
                centerWindow = floor(outputSize/2 + 1);
                HWDLocation     = centerX - centerWindow + 1;
            else
                % CropLocation is [x,y,z], which is [W,H,D]
                HWDLocation = CropLocation([2,1,3]);
            end
            topcrop = HWDLocation(1) - 1;
            bottomcrop = inputImageSize(1) - referenceImageSize(1) - topcrop;
            leftcrop = HWDLocation(2) - 1;
            rightcrop = inputImageSize(2) - referenceImageSize(2) - leftcrop;
            frontcrop = HWDLocation(3) - 1;
            backcrop = inputImageSize(3) - referenceImageSize(3) - frontcrop;

            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName(1), this.OutputTensorName,...
                "layers.Cropping3D", "cropping=((%d,%d),(%d,%d),(%d,%d))", {topcrop, bottomcrop, leftcrop, rightcrop, frontcrop, backcrop},...
                "", this.layerAnalyzer.IsTemporal);
        end
    end
end
