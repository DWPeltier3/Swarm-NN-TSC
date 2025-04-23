classdef TensorFlowRepository < nnet.internal.app.plugins.Repository
    % TensorFlowRepository  Holds TensorFlow Converter content for Deep Network Designer
    
    %   Copyright 2022-2023 The MathWorks, Inc.
    
    properties
        Layers = {
            nnet.internal.app.layer.ClipLayerTemplate();
            nnet.internal.app.layer.FlattenCStyleLayerTemplate();
            nnet.internal.app.layer.FlattenCStyleTFLayerTemplate();
            nnet.internal.app.layer.TFPreluLayerTemplate();
            nnet.internal.app.layer.TimeDistributedFlattenCStyleLayerTemplate();
            nnet.internal.app.layer.BinaryCrossEntropyRegressionLayerTemplate();
            nnet.internal.app.layer.ZeroPadding1dLayerTemplate();
            nnet.internal.app.layer.ZeroPadding2dLayerTemplate();
            nnet.internal.app.layer.SoftsignLayerTemplate();
            };
    end
end