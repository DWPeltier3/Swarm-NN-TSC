classdef ObjectLoaderResult < handle

%   Copyright 2022-2023 The MathWorks, Inc.

    properties
        APIType % Either Sequential, Functional, Subclass, Generic
        TranslationStrategy % Either nnet.internal.cnn.tensorflow.savedmodel.LoadFunctionalSequentialStrategy or nnet.internal.cnn.tensorflow.savedmodel.LoadSubClassedStrategy
        Class % ClassType
        Instance % instantiation of object 
        Children % children ObjectLoaders or Empty. 
        FunctionDef % optional FunctionDef for debugging 
        HasFcn % boolean signaling if the functiondef was found
        IsTopLevelLayer % boolean tells if current layer is the top level subclassed layer
        Namespace 
        LayerToOutName % if Functional or Sequential, this is a map that stores layernames to outputnames
        LayerToInName % if Functional or Sequential, this is a map that stores layernames to inputnames
        CodegeneratorObject % stores the codegenerator object used for writing and instantiating nested layers for subclassed models
        LayerInputEdges
        LayerOutputEdges
        LayerServingDefaultOutputs % if top level Subclassed layer, this will store the struct that maps outputs to output names
    end

    methods 
        function obj = ObjectLoaderResult()
            obj.APIType = '';
            obj.Class = '';
            obj.Instance = [];
            obj.Children = containers.Map;
            obj.IsTopLevelLayer = false;
        end
    end
end