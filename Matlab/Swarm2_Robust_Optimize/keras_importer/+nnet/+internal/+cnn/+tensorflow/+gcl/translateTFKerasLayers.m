function [lg, hasUnsupportedOp, supportsInputLayers, inputShapes] = translateTFKerasLayers(sm, packageName)
%
%   Copyright 2020-2023 The MathWorks, Inc.

import nnet.internal.cnn.tensorflow.*;
import nnet.internal.cnn.keras.util.*;									  
if isempty(sm.KerasManager.LayerGraph)
    throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:SubclassedModelNotSupported')));   
else
    lg = sm.KerasManager.LayerGraph; 
end

% get savedmodel folder name
s = what(sm.SavedModelPath);
p = strsplit(s(end).path, filesep);
smName = p{end};
hasUnsupportedOp = false;
hasSeqInputLayers = false;
inputShapes = {};
supportsInputLayers = true;
hasGCLDir = false;

% eliminate placeholder layers with custom layers. 
pl = gcl.placeholderLayerLabeler(lg, sm.KerasManager.AM.LayerSpecs); 
layerNameToSpecMap = createKerasLayerNameToSpecMap(sm.KerasManager.AM.LayerSpecs); 

if ~isempty(pl)
    % Generate the new folder and update path only if we need to create
    % custom layers. 
    
    if nargin < 2
        % if the custom layer path is not specified, we will use the
        % a default one in the saved model folder itself.
        packageName = smName;
        if ~(isvarname(packageName))
            packageName = matlab.lang.makeValidName(packageName);
            sm.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnspecifiedPackageName', MessageArgs={smName, packageName});            
        end
    else
        if ~(isvarname(packageName))
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:InvalidPackageNameSpecified', packageName, matlab.lang.makeValidName(packageName))));
        end
    end

    opFunctionsUsed = "";

    for i = 1:numel(pl)
        if ~(isa(pl(i),'nnet.keras.layer.PlaceholderInputLayer') || isa(pl(i),'nnet.keras.layer.PlaceholderOutputLayer'))
            % for each placeholder layer, locate the forward pass. 
            if isKey(sm.KerasManager.InternalTrackableGraph.LayerSpecToNodeIdx, pl(i).KerasConfiguration.name)
                nodestructidx = sm.KerasManager.InternalTrackableGraph.LayerSpecToNodeIdx(pl(i).KerasConfiguration.name); 
            elseif isKey(sm.KerasManager.InternalTrackableGraph.LayerSpecToNodeIdx, ['tf_op_layer_' pl(i).KerasConfiguration.name]) 
                nodestructidx = sm.KerasManager.InternalTrackableGraph.LayerSpecToNodeIdx(['tf_op_layer_' pl(i).KerasConfiguration.name]);
            else 
                continue;
            end
            
            nodestruct = sm.KerasManager.InternalTrackableGraph.NodeStruct{nodestructidx};
            fcn = getTopLevelFunction(sm, nodestruct);

            kerasType = gcl.util.iGetClassNameFromStruct(nodestruct);
            if strcmp(kerasType, 'Embedding')
                continue; 
            end
        
            if isempty(fcn)
                if strcmp(kerasType, 'TFOpLambda')
                    % Generate custom layer for TFOpLambda layer placeholder
                    % Create custom layer directory if it is required
                    [hasGCLDir, customLayerPath] = createCustomLayerDirectoryAndReturnPath(hasGCLDir, packageName, sm.ImportManager);
                    
                    TFOpLambdaPlaceHolderLayer = pl(i);                    
                    % Modify TFOpLambda function name for legalization
                    TFOpLambdaFcnName = ['k_' strrep(TFOpLambdaPlaceHolderLayer.KerasConfiguration.name, '.', '_')];
                    className = char(gcl.util.iMakeLegalMATLABNames({TFOpLambdaFcnName}));
                    classgenerator = nnet.internal.cnn.tensorflow.gcl.TFOpLambdaLayerTranslator(TFOpLambdaPlaceHolderLayer, customLayerPath, className, packageName, sm.ImportManager);
                    classgenerator.writeCustomLayer();
                    
                    opFunctionsUsed = [opFunctionsUsed classgenerator.OpFunctionsList]; %#ok<AGROW>
                    if classgenerator.HasUnsupportedOp
                        hasUnsupportedOp = true;
                    end
                    clear classgenerator
                    rehash
                    
                    % Instantiate the generated layer. Every generated layer
                    % constructor requires a name for the layer and the original Keras
                    % class name.
                    customLayerConstructor = str2func([packageName '.' className]);
                    newLayer = customLayerConstructor(pl(i).Name, kerasType);        
                    
                    % Replace placeholder layer with GCL
                    lg = lg.replaceLayer(pl(i).Name, newLayer);
                else
                    sm.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:TFGraphNotFound', MessageArgs={pl(i).Name});
                    continue;
                end
            else
                % Generate custom layer for Placeholder layer 
                layerInputShape = fcn.attr.x_input_shapes.list.shape(1).dim;
                if strcmp(kerasType, 'Dense') && numel(layerInputShape) == 2
                    translator = nnet.internal.cnn.keras.TranslatorForDenseLayer;
                    newLayer = translator.translate(layerNameToSpecMap(pl(i).Name), true, [] , []);
                    lg = lg.replaceLayer(pl(i).Name, [newLayer{:}]);
                    continue;
                end
    
                [allparamsNodeIdx, layerVariables] = getLayerVariables(sm, nodestruct.children, fcn);
    
                % Create custom layer directory if it is required
                [hasGCLDir, customLayerPath] = createCustomLayerDirectoryAndReturnPath(hasGCLDir, packageName, sm.ImportManager);
                
                % Generate custom layers
                className = ['k' char(fcn.Signature.legalname)];
                classgenerator = gcl.CustomLayerTranslator(pl(i), sm.GraphDef, fcn, layerVariables, customLayerPath, className, packageName, sm.ImportManager);
                [legalizedProperties, graphConstantNames] = classgenerator.writeCustomLayer();
                
                constants = classgenerator.Constants;
                opFunctionsUsed = [opFunctionsUsed classgenerator.OpFunctionsList]; %#ok<AGROW>
                if classgenerator.HasUnsupportedOp
                    hasUnsupportedOp = true;
                end
                clear classgenerator
                rehash
    
                % Instantiate the generated layer. Every generated layer
                % constructor requires a name for the layer and the original Keras
                % class name.
                customLayerConstructor = str2func([packageName '.' className]);
                newLayer = customLayerConstructor(pl(i).Name, kerasType);
    
                % Add learnable parameters to custom layer.
                for curPropertyIdx = 1:numel(allparamsNodeIdx)
                    curVal = layerVariables{curPropertyIdx}.Value;
                    newLayer.(legalizedProperties(curPropertyIdx)) = dlarray(curVal);
                end
    
                % Add constants to custom layer. 
                for curConstIdx = 1:numel(graphConstantNames)
                    newLayer.(graphConstantNames{curConstIdx}) = constants.lookupConst(graphConstantNames{curConstIdx});
                end
                % Replace placeholder layer with GCL
                lg = lg.replaceLayer(pl(i).Name, newLayer);
            end
        % Assign input layer for any placeholderInputLayers 
        elseif isa(pl(i),'nnet.keras.layer.PlaceholderInputLayer')
            [lg, isSupportedInputLayer, inputShapes, hasSeqInputLayers] = assignInputLayer(lg, pl(i), inputShapes, hasSeqInputLayers, sm.ImportManager);
            supportsInputLayers = supportsInputLayers & isSupportedInputLayer;
        end
    end

    % Remove timeDistributedLayers from dlnetworks
    if strcmp(sm.ImportManager.TargetNetwork, 'dlnetwork') && util.hasFoldingLayer(lg)
        idx = arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.SequenceFoldingLayer') || ...
            isa(layer, 'nnet.cnn.layer.SequenceUnfoldingLayer'), lg.Layers);

        timeDistributedLayers = lg.Layers(idx);
        names = arrayfun(@(layer) layer.Name, timeDistributedLayers, 'UniformOutput', false);
        lgWithoutTimeDistributed = removeLayers(lg, names);
        lg = layerGraph(lgWithoutTimeDistributed.Layers);
    end
        
    % Write to the custom layers package
    if hasGCLDir
        nnet.internal.cnn.tensorflow.util.writeOpFunctionScripts(opFunctionsUsed, customLayerPath, hasUnsupportedOp);
        rehash
    end
end
end

function [hasGCLDir, customLayerPath] = createCustomLayerDirectoryAndReturnPath(hasGCLDir, packageName, importManager)
    customLayerPath = [pwd filesep '+' packageName];
    if ~hasGCLDir        
        if isfolder(customLayerPath)
            importManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:PackageAlreadyExists', MessageArgs={packageName});
            rmdir(customLayerPath,'s');
        end
        mkdir(customLayerPath);
        hasGCLDir = true;
    end
end

function klntsm = createKerasLayerNameToSpecMap(KerasLayerSpecs)
    klntsm = containers.Map();
    for i = 1: numel(KerasLayerSpecs)
        klntsm(KerasLayerSpecs{i}.Name) = KerasLayerSpecs{i};
    end
end

function fcn = getTopLevelFunction(sm, nodestruct)
    % Get the TensorFlow graph that points to a layer's 'call' method
    fcn = [];
    children = nodestruct.children; 
    for childidx = 1:numel(children)
        child = children(childidx); 
        if strcmp(child.local_name, 'call_and_return_conditional_losses') || strcmp(child.local_name, 'call_and_return_all_conditional_losses')
            fcn = sm.KerasManager.InternalTrackableGraph.NodeStruct{child.node_id + 1}; 
            if isempty(fcn.function.concrete_functions)
                fcn = [];  
            else 
                fcn = fcn.function.concrete_functions{1}; 
            end
            fcn = sm.GraphDef.findFunction(fcn);
            break;
        end
    end
end

function [allparamsNodeIdx, layerVariables] = getLayerVariables(sm, children, fcn)
    import nnet.internal.cnn.tensorflow.*;
    % Getting learnables and non-learnables. 
    learnableNodeIdx = []; 
    for childidx = 1:numel(children)
        child = children(childidx); 
        if strcmp(child.local_name, 'trainable_variables')
            learnableNodes = sm.KerasManager.InternalTrackableGraph.NodeStruct{child.node_id + 1}.children;
            if ~isempty(learnableNodes)
                learnableNodeIdx = [learnableNodes.node_id] + 1;
            end
        end
    end

    concreteFcn = sm.KerasManager.InternalTrackableGraph.ConcreteFunctions.(matlab.lang.makeValidName(fcn.Signature.name)); 
    allparamsNodeIdx = concreteFcn.bound_inputs + 1; 
    layerVariables = cell(numel(allparamsNodeIdx), 1); 
    for j = 1:numel(allparamsNodeIdx)
        curLearnableStruct = sm.KerasManager.InternalTrackableGraph.NodeStruct{allparamsNodeIdx(j)}; 
        curVarName = char(curLearnableStruct.variable.name); 
        [curVar.Value, tensorndims] = tf2mex('checkpoint', ...
                                             [fullfile(sm.KerasManager.SavedModelPath, 'variables') filesep 'variables.index'], ...
                                             allparamsNodeIdx(j) - 1);
        curVar.IsLearnable = ismember(allparamsNodeIdx(j), learnableNodeIdx);
        curVar.ndims = tensorndims; 
        curVar.curVarName = curVarName; 
        layerVariables{j} = curVar; 
    end
end

% Replace PlaceholderInputLayer with a suitable DLT input layer (or None)
function [lg, supportsInputLayers, inputShapes, hasSeqInputLayers] = assignInputLayer(lg, pl, inputShapes, hasSeqInputLayers, importManager)

    % Assign input label for input label based on input rank and label of next layer
    pl = assignInputLabel(pl);

    % Replace None shape dimensions with 1 only if corresponding InputLabel is B or T
    formattedInputShape = formatInputShape(pl.batch_input_shape', pl.InputLabels{:});
    
    % If the input size still has any unknown dimensions we can't support
    % input layers
    supportsInputLayers = ~any(isnan(formattedInputShape));

    if supportsInputLayers
        % Calculate the difference in propagated label and rank to
        % determine number of extra batch dimensions
        diff = (pl.InputRank - numel(pl.OutputLabels{:}));
        
        % Image3dInputLayer
        if strcmp(pl.OutputLabels{:}, 'BSSSC') && pl.InputRank == 5
            lg = replaceLayer(lg, pl.Name, image3dInputLayer(formattedInputShape(2 + diff:end), 'Name', pl.Name, 'Normalization', 'none'));

        % ImageInputLayer
        elseif strcmp(pl.OutputLabels{:}, 'BSSC') && pl.InputRank == 4
            lg = replaceLayer(lg, pl.Name, imageInputLayer(formattedInputShape(2 + diff:end), 'Name', pl.Name, 'Normalization', 'none'));
    
        % SequenceInputLayers
        elseif strcmp(pl.OutputLabels{:}, 'BTC') && pl.InputRank == 3 && ~hasSeqInputLayers
            lg = replaceLayer(lg, pl.Name, sequenceInputLayer(formattedInputShape(end), 'Name', pl.Name, 'Normalization', 'none', 'MinLength', formattedInputShape(2 + diff)));
            % DLT only supports networks with 1 sequenceInputLayer
            hasSeqInputLayers = true;
        elseif strcmp(pl.OutputLabels{:}, 'BTSC') && pl.InputRank == 4 && ~hasSeqInputLayers
            lg = replaceLayer(lg, pl.Name, sequenceInputLayer(formattedInputShape(3:end), 'Name', pl.Name, 'Normalization', 'none', 'MinLength', formattedInputShape(2 + diff)));
            % DLT only supports networks with 1 sequenceInputLayer
            hasSeqInputLayers = true;
        elseif strcmp(pl.OutputLabels{:}, 'BTSSC') && pl.InputRank == 5 && ~hasSeqInputLayers
            lg = replaceLayer(lg, pl.Name, sequenceInputLayer(formattedInputShape(3:end), 'Name', pl.Name, 'Normalization', 'none', 'MinLength', formattedInputShape(2 + diff)));
            % DLT only supports networks with 1 sequenceInputLayer
            hasSeqInputLayers = true;
        elseif strcmp(pl.OutputLabels{:}, 'BTSSSC') && pl.InputRank == 6 && ~hasSeqInputLayers
            lg = replaceLayer(lg, pl.Name, sequenceInputLayer(formattedInputShape(3:end), 'Name', pl.Name, 'Normalization', 'none', 'MinLength', formattedInputShape(2 + diff)));
            % DLT only supports networks with 1 sequenceInputLayer
            hasSeqInputLayers = true;

        % FeatureInputLayer
        elseif strcmp(pl.OutputLabels{:}, 'BC') && pl.InputRank == 2
            lg = replaceLayer(lg, pl.Name, featureInputLayer(formattedInputShape(end), 'Name', pl.Name, 'Normalization', 'none'));
            
        % Generic inputLayer and InputVerificationLayer 
        % (for supporting all other possible formats)
        else
            % The New API supports Forward TensorFlow ordering for all
            % U-labeled data. Flip the order for the old API 
            if ~importManager.OnlySupportDlnetwork && all(pl.InputLabels{:} == 'U')
                formattedInputShape = fliplr(formattedInputShape);
            end

            % Initialize generic inputLayer with DLT size and format
            [labelsDLT, idx] = nnet.internal.cnn.tensorflow.util.sortToDLTLabel(pl.InputLabels{:});
            inputShapeDLT = formattedInputShape(idx);

            % inputLayer doesn't support inputs with trailing singleton U
            % dimensions
            if all(labelsDLT == 'U') && inputShapeDLT(end) == 1
                % Modify input layer name for inputVerificationLayer
                layerName = ['verify_' pl.Name];
                lg = replaceLayer(lg, pl.Name, nnet.keras.layer.inputVerificationLayer(layerName, inputShapeDLT, labelsDLT));
                % Changing flag to false since this isn't a DLT input layer
                supportsInputLayers = false;
            else
                lg = replaceLayer(lg, pl.Name, inputLayer(inputShapeDLT, labelsDLT, 'Name', pl.Name));
                if strcmp(importManager.TargetNetwork, 'dagnetwork')
                    % DAGNetworks don't support the generic inputLayer
                    supportsInputLayers = false;
                end
            end

        end
    else
        % Warn about partially missing input size information
        importManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:InputWithNaNs', MessageArgs={pl.Name});        
        
        % Remove the placeholder input layer
        lg = removeLayers(lg, pl.Name);
    end
    
    % Assign Input Size(s) and Format(s) to inputShapes
    inputShapes{end+1} =  [formattedInputShape, pl.InputLabels];

end

function pl = assignInputLabel(pl)
    pl = updatePILOutputLabels(pl);

    if pl.InputRank == numel(pl.OutputLabels{:})
        pl.InputLabels = pl.OutputLabels;
    % Consider extra dimensions as folded batch dimensions and label them
    % with 'U's
    elseif pl.InputRank > numel(pl.OutputLabels{:}) && ~isempty(pl.OutputLabels{:})
        augULabels = char('U' + zeros(1, (pl.InputRank - numel(pl.OutputLabels{:}))));
        pl.InputLabels = {['B' augULabels pl.OutputLabels{:}(2:end)]};
    % Assign all 'U' labels if labels for the input can't be propagated
    elseif isempty(pl.OutputLabels{:})
        % MATLAB doesn't support input rank < 2
        % Assign a singleton 'U' label for the missing dimension
        if pl.InputRank < 2
            pl.InputLabels = {'UU'};
            pl.batch_input_shape(end+1) = 1;
            pl.batch_input_shape = flip(pl.batch_input_shape');
        else
            pl.InputLabels = {[char('U' + zeros(1, pl.InputRank))]};
        end
    else
        % If InputRank is less than Label, then increase rank and assign
        % singleton. This is the case with Embedding Layer
        pl.InputRank = pl.InputRank + 1;
        pl.batch_input_shape(end+1) = 1;
        pl.InputLabels = pl.OutputLabels;
        % warning('Unable to label the output of layer:');
    end
end

% Convert TF input shape into DLT compatible input shape
function inputShape = formatInputShape(inputShape, inputLabels)
    for i=1:numel(inputShape)
        if isnan(inputShape(i)) && ismember(inputLabels(i), {'B', 'T'})
            inputShape(i) = 1;
        end
    end
    % For forward TF format with all U labels treat first dimension as batch
    if isnan(inputShape(1)) && strcmp(inputLabels, char('U' + zeros(1, numel(inputShape))))
        inputShape(1) = 1;
    end
end

% Interpreting labels propagated to the input layer
function pil = updatePILOutputLabels(pil)
    dlOutRank = pil.InputRank;
    OutputLabels = pil.OutputLabels;
    for i = 1:numel(OutputLabels)
        fwdTFLabel = OutputLabels{i};
            if strcmp(fwdTFLabel,'fcs')
            % Output labels not found due to a blocking FlattenCStyleLayer
            % BSS* case
                trailingSLabels = char('S' + zeros(1, dlOutRank - 1));
                fwdTFLabel = ['B' trailingSLabels];
            elseif strcmp(fwdTFLabel,'fcst')
            % Output labels not found due to a blocking FlattenCStyleLayer
            % BTSS* case with 
                trailingSLabels = char('S' + zeros(1, dlOutRank - 2));
                fwdTFLabel = ['BT' trailingSLabels];
            elseif strcmp(fwdTFLabel,'sm')
            % Output labels not found due to a softmax layer
            % BU*C case
                midULabels = char('U' + zeros(1, dlOutRank - 2));
                fwdTFLabel = ['B' midULabels 'C']; 
            elseif strcmp(fwdTFLabel,'smt')
            % Output labels not found due to a softmax layer
            % BTU*C case
                midULabels = char('U' + zeros(1, dlOutRank - 3));
                fwdTFLabel = ['BT' midULabels 'C']; 
            elseif strcmp(fwdTFLabel,'classregout')
            % Output labels not found due to a regression/classification output layer
            % BU*C case
                midULabels = char('U' + zeros(1, dlOutRank - 2));
                fwdTFLabel = ['B' midULabels 'C']; 
            elseif strcmp(fwdTFLabel,'pixelout') %|| strcmp(fwdTFLabel,'bce')
            % Output labels not found due to a pixelclassification 
            % or BinaryCrossEntropyRegressionLayer output layer BC / BSSC / BSSSC case
                midSLabels = char('S' + zeros(1, dlOutRank - 2));
                fwdTFLabel = ['B' midSLabels 'C'];
            elseif strcmp(fwdTFLabel,'t') || strcmp(fwdTFLabel, 'tdo')
            % Output labels not found due to a blocking
            % TimeDistributedIn/Out Layer
            % BTS*C case
                switch dlOutRank
                    case 3
                        fwdTFLabel = 'BTC';
                    case 4
                        fwdTFLabel = 'BTSC';
                    case 5
                        fwdTFLabel = 'BTSSC';
                    case 6
                        fwdTFLabel = 'BTSSSC';
                    otherwise
                        fwdTFLabel = '';
                end
            end
    OutputLabels{i} = fwdTFLabel;
    end
    pil.OutputLabels = OutputLabels;
end

