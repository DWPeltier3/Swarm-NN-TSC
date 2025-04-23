function Network = importNetworkFromTensorFlow(modelFolder, options)
% Copyright 2022-2023 The MathWorks, Inc.

arguments
    modelFolder {mustBeFolder}
    options.PackageName {mustBeTextScalar} = ''
    options.Namespace {mustBeTextScalar} = ''
end

%% Call to importTensorFlowNetwork
Network = nnet.internal.cnn.tensorflow.importTensorFlowNetwork(modelFolder, 'TargetNetwork', 'dlnetwork', 'PackageName', options.PackageName, 'Namespace',options.Namespace, 'OnlySupportDlnetwork', true);
