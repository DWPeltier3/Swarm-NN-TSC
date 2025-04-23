function labeledPlaceholderLayers = placeholderLayerLabeler(LayerGraph, KerasLayerSpecs)
% Copyright 2022-2023 The MathWorks, Inc.

if isa(LayerGraph, 'nnet.cnn.LayerGraph')
    Layers = LayerGraph.Layers;
else
    Layers = LayerGraph;
end

layerNameToTypeMap = createKerasLayerNameToTypeMap(KerasLayerSpecs);
lgDiGraph = LayerGraph.extractPrivateDirectedGraph;

% kerasLayerLabelMap holds the keras layers and their input/output formats in forward TF order
kerasLayerLabelMap = createKerasLayerLabelMap();

[labeledPlaceholderLayers, lplIndices] = findPlaceholderLayers(Layers);
lplIndices = lplIndices';
if any(lplIndices)
    for i = lplIndices(end:-1:1)   
        Layers(i) = assignLabels(i, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap); 
    end
    [labeledPlaceholderLayers, ~] = findPlaceholderLayers(Layers);
end
end

function labelledPL = assignLabels(lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap)
    labelledPL = lgDiGraph.Nodes.Layers(lIdx);
    % Assign output labels
    for j = 1:labelledPL.NumOutputs
        if isempty(labelledPL.OutputLabels{j})
            labelFound = findLabelInSuccessors(lIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap, j);            
        end
        if ismember(labelFound,{'*', '-'}) 
            % label not found leave it empty to return all U's
            labelledPL.OutputLabels{j} = '';
        else
            % label found
            labelledPL.OutputLabels{j} = labelFound;
        end
    end
end

function labelFound = findLabelInSuccessors(currIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap, outputNum)
    
    if nargin == 6
        succIndices = [];
        edges = find(lgDiGraph.Edges.EndNodes(:,1)==currIdx);
        for i = 1:numel(edges)
            edge = edges(i);
            endportsMat = cell2mat(lgDiGraph.Edges.EndPorts(edges));
            if size(endportsMat,1) > 1
                % parallel edges
                if any(endportsMat(:,1) == outputNum)
                    succIndices(end+1) = lgDiGraph.Edges.EndNodes(edge,2); %#ok<AGROW>
                end
            else
                if endportsMat(:,1) == outputNum
                    succIndices(end+1) = lgDiGraph.Edges.EndNodes(edge,2); %#ok<AGROW>
                end
            end
        end
    else
        succIndices = lgDiGraph.successors(currIdx);
    end

    allSuccLabels = {};
    if ~isempty(succIndices)        
        for i = 1:numel(succIndices)
            succIdx = succIndices(i);
            allSuccLabels{end+1} = getLabelFromOneSuccessor(succIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap); %#ok<AGROW> 
        end
    end
    
    hasTDO = cellfun(@(x) any(ismember(x,{'tdo'})), allSuccLabels, 'UniformOutput', true);
    if numel(allSuccLabels) == 2 && any(hasTDO)
       allSuccLabelsWoTDO = allSuccLabels(~hasTDO);
       if ~isempty(allSuccLabelsWoTDO) && ismember(allSuccLabelsWoTDO{end},{'*'})
           allSuccLabels = {'t'};
       end
    end

    validSuccLabels = setdiff(allSuccLabels,{'*','-',''});
    if ~isempty(validSuccLabels)
        hasTDO = cellfun(@(x) any(ismember(x,{'tdo'})), validSuccLabels, 'UniformOutput', true);        
        if numel(validSuccLabels) == 2
            % ignore any 'tdo' labels, coming from sequence unfolding layers            
            if any(hasTDO)
                validSuccLabels = validSuccLabels(~hasTDO);
                % add T dimension
                validSuccLabels{end} = addTemporalDim(validSuccLabels{end});
            end
        end
        % pick the last valid label
        labelFound = validSuccLabels{end};
    else
        labelFound = '';
    end
end

function temporalLabel = addTemporalDim(nonTLabel)    
    if contains(nonTLabel,{'B','S','C'}) && ~contains(nonTLabel,{'T'})
        temporalLabel = [nonTLabel(1) 'T' nonTLabel(2:end)];
    elseif ismember(nonTLabel,{'sm', 'fcs'})
        temporalLabel = [nonTLabel 't'];
    elseif strcmp(nonTLabel, 'BTC')
        temporalLabel = 'BTSC';
    end
end

function label = getLabelFromOneSuccessor(succIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap)
    label = '*';
    succL = lgDiGraph.Nodes.Layers(succIdx);
    succLName = matlab.lang.makeValidName(succL.Name);

    if ismember(class(lgDiGraph.Nodes.Layers(lIdx)), {'nnet.keras.layer.PlaceholderInputLayer'}) && lgDiGraph.Nodes.Layers(lIdx).InputRank == 2 && ismember(layerNameToTypeMap(succLName), {'Dense'})
           label = 'BC';
    elseif isKey(layerNameToTypeMap, succLName) && ~isa(succL,'nnet.cnn.layer.PlaceholderLayer') && isKey(kerasLayerLabelMap,layerNameToTypeMap(succLName))
       if numel(lgDiGraph.predecessors(succIdx)) > 1 && ismember(layerNameToTypeMap(succLName), {'Add', 'Merge', 'Multiply'})
           % successor is a multi-input layer {'Add', 'Concatenate', 'Merge', 'Multiply'} that has data format constraints
           % hence search other predecessors of this successor layer for a data format
           for i = 1:succL.NumOutputs
                label = findLabelInSuccessors(succIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap, i);
           end
           if isempty(label)
                label = findLabelInPredecessors(succIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap);
           end
       elseif ismember(layerNameToTypeMap(succLName), {'Activation'}) && ismember(class(succL),{'nnet.cnn.layer.SoftmaxLayer'})
           % successor is a softmax layer that requires channels to be labelled.
           label = 'sm';
       else
           succLLabelsStruct = kerasLayerLabelMap(layerNameToTypeMap(succLName));
           label = succLLabelsStruct.InputLabels;
       end
    elseif isKey(layerNameToTypeMap, succLName) && isa(succL,'nnet.cnn.layer.PlaceholderLayer') && ismember(layerNameToTypeMap(succLName), {'Dense'})
            if numel(lgDiGraph.predecessors(succIdx)) > 1 && ismember(layerNameToTypeMap(succLName), {'Add', 'Merge', 'Multiply'})
               % successor is a multi-input layer {'Add', 'Concatenate', 'Merge', 'Multiply'} that has data format constraints
               % hence search other predecessors of this successor layer for a data format
               label = findLabelInPredecessors(succIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap);
           elseif ismember(layerNameToTypeMap(succLName), {'Activation'}) && ismember(class(succL),{'nnet.cnn.layer.SoftmaxLayer'})
               % successor is a softmax layer that requires channels to be labelled.
               label = 'sm';
           else
               succLLabelsStruct = kerasLayerLabelMap(layerNameToTypeMap(succLName));
               label = succLLabelsStruct.InputLabels;
           end
    elseif isKey(layerNameToTypeMap, succLName) && isa(succL,'nnet.cnn.layer.PlaceholderLayer') && ismember(layerNameToTypeMap(succLName), {'Add', 'Concatenate', 'Merge', 'Multiply'})
           succLLabelsStruct = kerasLayerLabelMap(layerNameToTypeMap(succLName));
           label = succLLabelsStruct.InputLabels;    
    elseif ismember(class(succL), {'nnet.cnn.layer.RegressionOutputLayer','nnet.cnn.layer.ClassificationOutputLayer'})
           label = 'classregout';												 
    elseif ismember(class(succL), {'nnet.cnn.layer.PixelClassificationLayer'})
           label = 'pixelout';   
    else
       % unrecognized successor layer / placeholder layer / blocker layer
       return;
    end
    
    if strcmp(label,'*')
        % keep searching successors
        label = findLabelInSuccessors(succIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap);    
    end
end        

function labelFound = findLabelInPredecessors(currIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap)
    preds = lgDiGraph.predecessors(currIdx);
    predIndices = setdiff(preds,lIdx);
    allPredLabels = {};
    if ~isempty(predIndices)        
        for i = 1:numel(predIndices)
            predIdx = predIndices(i);
            allPredLabels{end+1} = getLabelFromOnePredecessor(predIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap); %#ok<AGROW> 
        end
    end
    
    validPredLabels = setdiff(allPredLabels,{'*','-',''});
    if ~isempty(validPredLabels)
        % pick the last valid label
        labelFound = validPredLabels{end};
    else
        labelFound = '';
    end
end

function label = getLabelFromOnePredecessor(predIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap)
    label = '*';
    predL = lgDiGraph.Nodes.Layers(predIdx);
    if isKey(layerNameToTypeMap, predL.Name) && ~isa(predL,'nnet.cnn.layer.PlaceholderLayer') && isKey(kerasLayerLabelMap,layerNameToTypeMap(predL.Name))
       if numel(lgDiGraph.predecessors(predIdx)) > 1
           % predecessor is a multi-input layer that has data format constraints
           % hence search other predecessors of this successor layer for a data format
           label = findLabelInPredecessors(predIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap);
       else
           predLLabelsStruct = kerasLayerLabelMap(layerNameToTypeMap(predL.Name));
           label = predLLabelsStruct.OutputLabels;
       end
    elseif isKey(layerNameToTypeMap, predL.Name) && isa(predL,'nnet.cnn.layer.PlaceholderLayer') && ...
            isKey(kerasLayerLabelMap,layerNameToTypeMap(predL.Name)) && ismember(layerNameToTypeMap(predL.Name), {'Lambda'})
        % if a Lambda layer is found in predecessor traversal then don't
        % halt search. Instead search for labels in its next successor
        succIndices = lgDiGraph.successors(predIdx);
        succIdx = setdiff(succIndices, lgDiGraph.successors(lIdx));
        label = getLabelFromOneSuccessor(succIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap);
    else
       % unrecognized predecessor layer / placeholder layer / blocker layer
       return;
    end
    
    if strcmp(label,'*')
        % keep searching predecessors
        label = findLabelInPredecessors(predIdx, lIdx, lgDiGraph, kerasLayerLabelMap, layerNameToTypeMap);
    end
end

function kllm = createKerasLayerLabelMap()
    kllm = containers.Map();  
    kllm('Activation') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('Add') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('AveragePooling1D') = struct('InputLabels', 'BTC', 'OutputLabels', 'BTC');
    kllm('AveragePooling2D') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('AveragePooling3D') = struct('InputLabels', 'BSSSC', 'OutputLabels', 'BSSSC');
    kllm('BatchNormalization') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('Bidirectional') = struct('InputLabels', 'BTC', 'OutputLabels', '*');
    kllm('Concatenate') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('Conv1D') = struct('InputLabels', 'BTC', 'OutputLabels', 'BTC');
    kllm('Conv2D') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('Conv3D') = struct('InputLabels', 'BSSSC', 'OutputLabels', 'BSSSC');
    kllm('Conv2DTranspose') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('Conv3DTranspose') = struct('InputLabels', 'BSSSC', 'OutputLabels', 'BSSSC');
    kllm('CuDNNGRU') = struct('InputLabels', 'BTC', 'OutputLabels', 'BC/BTC');
    kllm('Dense') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('DepthwiseConv2D') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('Dropout') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('ELU') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('Flatten') = struct('InputLabels', 'fcs', 'OutputLabels', 'BC');
    kllm('GlobalAveragePooling1D') = struct('InputLabels', 'BTC', 'OutputLabels', 'BTC');
    kllm('GlobalAveragePooling2D') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('GlobalAveragePooling3D') = struct('InputLabels', 'BSSSC', 'OutputLabels', 'BSSSC');
    kllm('GlobalMaxPooling1D') = struct('InputLabels', 'BTC', 'OutputLabels', 'BTC');
    kllm('GlobalMaxPooling2D') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('GlobalMaxPooling3D') = struct('InputLabels', 'BSSSC', 'OutputLabels', 'BSSSC');
    kllm('GRU') = struct('InputLabels', 'BTC', 'OutputLabels', 'BC/BTC');
    kllm('InputLayer') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('LeakyReLU') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('Lambda') = struct('InputLabels', '-', 'OutputLabels', '-');
    kllm('LSTM') = struct('InputLabels', 'BTC', 'OutputLabels', 'BC/BTC');
    kllm('CuDNNLSTM') = struct('InputLabels', 'BTC', 'OutputLabels', 'BC/BTC');
    kllm('MaxPooling1D') = struct('InputLabels', 'BTC', 'OutputLabels', 'BTC');
    kllm('MaxPooling2D') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('MaxPooling3D') = struct('InputLabels', 'BSSSC', 'OutputLabels', 'BSSSC');
    kllm('Merge') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('Multiply') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('PReLU') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('ReLU') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('Reshape') = struct('InputLabels', '-', 'OutputLabels', '-');
    kllm('SeparableConv2D') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('Softmax') = struct('InputLabels', 'sm', 'OutputLabels', '*');
    kllm('ZeroPadding1D') = struct('InputLabels', 'BTC', 'OutputLabels', 'BTC');
    kllm('ZeroPadding2D') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('TimeDistributedIn') = struct('InputLabels', '*', 'OutputLabels', '*');
    kllm('TimeDistributedOut') = struct('InputLabels', 'tdo', 'OutputLabels', '*');
    kllm('UpSampling1D') = struct('InputLabels', 'BTC', 'OutputLabels', 'BTC');
    kllm('Upsampling2D') = struct('InputLabels', 'BSSC', 'OutputLabels', 'BSSC');
    kllm('Embedding') = struct('InputLabels', 'BTC', 'OutputLabels', 'BTC');
end

function klntm = createKerasLayerNameToTypeMap(KerasLayerSpecs)
    klntm = containers.Map();
    for i = 1: numel(KerasLayerSpecs)
        klntm(matlab.lang.makeValidName(KerasLayerSpecs{i}.Name)) = KerasLayerSpecs{i}.Type;
    end
end
