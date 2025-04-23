function [lg, hasUnsupportedOp] = translateTFKerasLayersByName_v2(InternalTrackableGraph, GraphDef, layerGraph, SavedModelPath, importManager, layerSpecs, packageName)
% Translates Keras Layers found in Functional/Sequential models inside
% subclassed models

%   Copyright 2022-2023 The MathWorks, Inc.

import nnet.internal.cnn.tensorflow.*;
import nnet.internal.cnn.keras.util.*;                                                                    
lg = layerGraph;

% get savedmodel folder name
s = what(SavedModelPath);
p = strsplit(s(end).path, filesep);
smName = p{end};
hasUnsupportedOp = false;

% update the placeholder layer's input and output labels using the
% layer-level labeler
pl = gcl.placeholderLayerLabeler(lg, layerSpecs);

% eliminate placeholder layers with custom layers. 
if ~isempty(pl)
    % Generate the new folder and update path only if we need to create
    % custom layers. 
    
    if nargin < 7
        % if the custom layer path is not specified, we will use the
        % a default one in the saved model folder itself.
        packageName = smName;
        if ~(isvarname(packageName))
            packageName = matlab.lang.makeValidName(packageName);                     
        end
    else
        if ~(isvarname(packageName))
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:InvalidPackageNameSpecified',packageName,matlab.lang.makeValidName(packageName))));
        end
    end
    

    customLayerPath = [pwd filesep '+' packageName];
    if ~exist(customLayerPath, 'dir')
                % A submodel could have already created this package containing GCLs
                % Create this package only if it doesn't already exist 
                mkdir(customLayerPath);
    end
    opFunctionsUsed = "";

    for i = 1:numel(pl)
        if isa(pl(i),'nnet.keras.layer.PlaceholderInputLayer') || isa(pl(i),'nnet.keras.layer.PlaceholderOutputLayer')
            lg = removeLayers(lg, pl(i).Name);
        else
            % for each placeholder layer, locate the forward pass. 
            if isKey(InternalTrackableGraph.LayerSpecToNodeIdx, pl(i).KerasConfiguration.name)
                nodestructidx = InternalTrackableGraph.LayerSpecToNodeIdx(pl(i).KerasConfiguration.name); 
            elseif isKey(InternalTrackableGraph.LayerSpecToNodeIdx, ['tf_op_layer_' pl(i).KerasConfiguration.name]) 
                nodestructidx = InternalTrackableGraph.LayerSpecToNodeIdx(['tf_op_layer_' pl(i).KerasConfiguration.name]);
            else
                continue; 
            end
            
            nodestruct = InternalTrackableGraph.NodeStruct{nodestructidx};
            kerasType = gcl.util.iGetClassNameFromStruct(nodestruct);
            if strcmp(kerasType, 'Embedding')
                continue; 
            end 
            
            if strcmp(pl(i).Type,'TensorFlowOpLayer')
               fcn = getFunctionForTensorFlowOpLayer(pl(i));
               layerVariables = {};
               allparamsNodeIdx = {};
            else
               fcn = getTopLevelFunction(InternalTrackableGraph, GraphDef, nodestruct); 
               if isempty(fcn)
                    importManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:TFGraphNotFound', MessageArgs={pl(i).Name});
                    continue;
               else
                    [allparamsNodeIdx, layerVariables] = getLayerVariables_v2(SavedModelPath, InternalTrackableGraph, nodestruct.children, fcn);
               end
            end                      

            % Generate custom layers
            className = ['k' char(fcn.Signature.legalname)];
            classgenerator = gcl.CustomLayerTranslator(pl(i), GraphDef, fcn, layerVariables, customLayerPath, className, packageName, importManager);
            
            % Set InSubclassed = true since this custom layer is generated from a subclassed model
            classgenerator.InSubclassed = true;
            [legalizedProperties, graphConstantNames] = classgenerator.writeCustomLayer();
            
            % Don't replace the placeholder layer if 
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

            lg = lg.replaceLayer(pl(i).Name, newLayer);
        end
    end
    nnet.internal.cnn.tensorflow.util.writeOpFunctionScripts(opFunctionsUsed, customLayerPath, hasUnsupportedOp);
    rehash
end
end

function fcn = getFunctionForTensorFlowOpLayer(placeholderLayer)
    % Get a synthetic TF function graph for this single Op TF Layer
    fcnObj.signature = makefcnSignature(placeholderLayer);
    for i = 1:numel(placeholderLayer.InputNames)
        fcnObj.attr.x_input_shapes.list.shape(i).dim = [];
        fcnObj.attr.x_input_shapes.list.shape(i).unknown_rank = 1;
        placeholderLayer.KerasConfiguration.node_def.input{i} = placeholderLayer.InputNames{i};
    end
    fcnObj.arg_attr = []; 
    fcnObj.resource_arg_unique_id = []; 
    placeholderLayer.KerasConfiguration.node_def.device = '';
    fcnObj.node_def = nnet.internal.cnn.tensorflow.savedmodel.TFNodeDef(placeholderLayer.KerasConfiguration.node_def);
    
    for i = 1:numel(placeholderLayer.OutputNames)
        fcnObj.ret.(placeholderLayer.OutputNames{i}) = [placeholderLayer.KerasConfiguration.name ':output:' num2str(i)];
    end
    fcnObj.control_ret = []; 
    fcn =  nnet.internal.cnn.tensorflow.savedmodel.TFFunction(fcnObj);
end

function opdef = makefcnSignature(placeholderLayer)
    opdef.name = placeholderLayer.KerasConfiguration.name;
    inputArgsStruct = struct;
    for i = 1:numel(placeholderLayer.InputNames)
        inputArgsStruct(i).name = placeholderLayer.InputNames{i};
        inputArgsStruct(i).description = [];
        inputArgsStruct(i).type = 'DT_FLOAT';
        inputArgsStruct(i).type_attr = [];
        inputArgsStruct(i).number_attr = [];
        inputArgsStruct(i).type_list_attr = [];
        inputArgsStruct(i).is_ref = 0;
    end
    outputArgsStruct = struct;
    for i = 1:numel(placeholderLayer.OutputNames)
        outputArgsStruct(i).name = placeholderLayer.OutputNames{i};
        outputArgsStruct(i).description = [];
        outputArgsStruct(i).type = 'DT_FLOAT';
        outputArgsStruct(i).type_attr = [];
        outputArgsStruct(i).number_attr = [];
        outputArgsStruct(i).type_list_attr = [];
        outputArgsStruct(i).is_ref = 0;
    end
    opdef.input_arg = inputArgsStruct; 
    opdef.output_arg = outputArgsStruct; 
    opdef.control_output = []; 
    opdef.attr = placeholderLayer.KerasConfiguration.node_def.attr; 
    opdef.summary = []; 
    opdef.description = []; 
    opdef.is_commutative = []; 
    opdef.is_aggregate = []; 
    opdef.is_stateful = []; % Maybe this can be set from the LayerSpec
    opdef.allows_uninitialized_input = [];    
end

function fcn = getTopLevelFunction(InternalTrackableGraph, GraphDef, nodestruct)
    % Get the TensorFlow graph that points to a layer's 'call' method
    fcn = [];
    children = nodestruct.children; 
    for childidx = 1:numel(children)
        child = children(childidx); 
        if strcmp(child.local_name, 'call_and_return_conditional_losses') || strcmp(child.local_name, 'call_and_return_all_conditional_losses')
            fcn = InternalTrackableGraph.NodeStruct{child.node_id + 1}; 
            if isempty(fcn.function.concrete_functions)
                fcn = [];  
            else 
                fcn = fcn.function.concrete_functions{1}; 
            end
            fcn = GraphDef.findFunction(fcn);
            break;
        end
    end
end

function [allparamsNodeIdx, layerVariables] = getLayerVariables_v2(SavedModelPath, InternalTrackableGraph, children, fcn)
    import nnet.internal.cnn.tensorflow.*;
    % Getting learnables and non-learnables. 
    learnableNodeIdx = []; 
    for childidx = 1:numel(children)
        child = children(childidx); 
        if strcmp(child.local_name, 'trainable_variables')
            learnableNodes = InternalTrackableGraph.NodeStruct{child.node_id + 1}.children;
            if ~isempty(learnableNodes)
                learnableNodeIdx = [learnableNodes.node_id] + 1;
            end
        end
    end

    concreteFcn = InternalTrackableGraph.ConcreteFunctions.(matlab.lang.makeValidName(fcn.Signature.name)); 
    allparamsNodeIdx = concreteFcn.bound_inputs + 1; 
    layerVariables = cell(numel(allparamsNodeIdx), 1); 
    for j = 1:numel(allparamsNodeIdx)
        curLearnableStruct = InternalTrackableGraph.NodeStruct{allparamsNodeIdx(j)}; 
        if ~isfield(curLearnableStruct, 'variable')

            continue
        end 
        curVarName = char(curLearnableStruct.variable.name); 
        [curVar.Value, tensorndims] = tf2mex('checkpoint', ...
                                             [fullfile(SavedModelPath, 'variables') filesep 'variables.index'], ...
                                             allparamsNodeIdx(j) - 1);
        curVar.IsLearnable = ismember(allparamsNodeIdx(j), learnableNodeIdx);
        curVar.ndims = tensorndims; 
        curVar.curVarName = curVarName; 
        layerVariables{j} = curVar; 
    end
    % remove variables which arent in the checkpoint file. 
    emptyVars = cellfun(@(x)isempty(x), layerVariables); 
    layerVariables(emptyVars) = []; 
end
