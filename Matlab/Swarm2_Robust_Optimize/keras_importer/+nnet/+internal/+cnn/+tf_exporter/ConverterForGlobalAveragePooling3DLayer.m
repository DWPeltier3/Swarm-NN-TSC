classdef ConverterForGlobalAveragePooling3DLayer < nnet.internal.cnn.tf_exporter.LayerConverter

    %   Copyright 2022 The MathWorks, Inc.

    % Supported MATLAB input formats and behavior:
    %     SSSCB --> SSSCB = 111CB
    %     SSSCBT --> SSSCBT = 111CBT
    %     SSSCT --> SSSCT = 111CT
    %     SSCBT --> SSCB = 11CB      (spatio-temporal)
    methods
        function convertedLayer = toTensorflow(this)
            convertedLayer = nnet.internal.cnn.tf_exporter.ConvertedLayer;
            if this.InputFormat=="SSCBT"
                % In TF the desired mapping is BTSSC --> BSSC = B11C. To do
                % this, we do GAP3d to obtain BC, then call
                % layers.Reshape((1,1,-1)), which means C should be
                % reshaped to 11C.
                convertedLayer.layerCode = [
                    kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.GlobalAveragePooling3D", "", {}, "", false)

                    kerasCodeLine(this, this.OutputTensorName, this.OutputTensorName,...
                    "layers.Reshape", "(1,1,-1)", {}, "", false)
                    ];
            else
                % Just do a normal GAP, possiblt TimeDistributed
                convertedLayer.layerCode = kerasCodeLine(this, this.InputTensorName, this.OutputTensorName,...
                    "layers.GlobalAveragePooling3D", "keepdims=True", {}, "", this.layerAnalyzer.IsTemporal);
            end
        end
    end
end