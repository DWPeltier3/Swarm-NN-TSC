function setAdditionalResourceLocation()

% Copyright 2022 The MathWorks, Inc.

m = message('nnet_cnn_onnx:onnx:Nargs');
try
    m.getString();
catch
    rootDir = nnet.internal.cnn.tf_exporter.supportPackageRootDir();
    matlab.internal.msgcat.setAdditionalResourceLocation(rootDir)
end
end