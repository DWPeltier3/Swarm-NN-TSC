function tf = hasFoldingLayer(LayersOrGraph)
    % Copyright 2021 The MathWorks, Inc.    
    isLG = isa(LayersOrGraph, 'nnet.cnn.LayerGraph');     
    if isLG
        Layers = LayersOrGraph.Layers; 
    else 
        Layers = LayersOrGraph; 
    end
    tf = any(arrayfun(@(L)isa(L, 'nnet.cnn.layer.SequenceFoldingLayer'), Layers)); 
end 