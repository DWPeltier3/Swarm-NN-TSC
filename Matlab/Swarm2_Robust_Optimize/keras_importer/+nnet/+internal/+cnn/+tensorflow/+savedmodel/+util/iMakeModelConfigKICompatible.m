function updatedMdlCfg = iMakeModelConfigKICompatible(mdlConfig, iTGraph, lastSeqModelName)
%   This function updates the Keras model configuration struct so that it 
%   becomes compatible with Keras Importer, i.e., Keras Importer 
%   is able to translate the model config into DLT layers 

%   Copyright 2020-2023 The MathWorks, Inc.
    mdlConfig = removeExtraneousFields(mdlConfig);
    updatedMdlCfg = updateModelConfig(mdlConfig, iTGraph, lastSeqModelName);
end

function mdlConfig = removeExtraneousFields(mdlConfig)
% Helper function that removes unwanted fields from the model configuration struct
requiredFields = {'name','trainable','expects_training_arg','dtype',...
                    'batch_input_shape','must_restore_from_config','class_name',...
                    'config','shared_object_id','input_spec','build_input_shape',...
                    'is_graph_network','keras_version',...
                    'backend','model_config','training_config'};

actualFields = fieldnames(mdlConfig);
extraFields = setdiff(actualFields,requiredFields);
if ~isempty(extraFields)
% Remove any extraneous fields that are structs or cell arrays
% as these can cause side-effects during updateModelConfig
    for f = 1:numel(extraFields)
        field = extraFields{f};
        if iscell(mdlConfig.(field)) || isstruct(mdlConfig.(field))
            mdlConfig = rmfield(mdlConfig,extraFields(f));
        end
    end
end
end

function s1 = updateModelConfig(s, iTGraph, lastSeqModelName)
% Helper function that recursively updates the model config struct
    import nnet.internal.cnn.tensorflow.*;
    for i = 1:length(s)
        if(isstruct(s(i)))
            fn = fieldnames(s);
            if isfield(s(i),'class_name') && strcmp(s(i).class_name,'Sequential') && isfield(s(i),'name')
                % Keep track of the last sequential model name
                lastSeqModelName = s(i).name;
            end
            for j=1:numel(fn)
                if strcmp(fn{j},'layers')
                    if isfield(s(i).(fn{j}),'config')
                        for l = 1:numel(s(i).(fn{j}))
                            cfg = s(i).(fn{j})(l).config;
                            if ~isfield(cfg,'name') && ~isfield(cfg,'layerWasSavedWithoutConfig')
                                s(i).(fn{j})(l).config.layerWasSavedWithoutConfig = 1;                            
                            end
                        end
                    end
                end
                if (strcmp(fn{j},'class_name') && ...
                        (strcmp(s(i).(fn{j}),'__tuple__')||strcmp(s(i).(fn{j}),'TensorShape')) ...
                        && any(ismember(fn, 'items')))

                    if isstruct(s(i).items) && length(s(i).items) > 1
                        % handle special case where 'items' is a struct
                        % e.g. for padding layer
                        for k = 1:length(s(i).items)
                            tempMat(k,:) = s(i).items(k).items; %#ok<AGROW>
                        end
                        s.items = tempMat;
                    end
                    
                    if length(s) > 1
                        % handle special case for multiple-inputs
                        for k = 1:length(s)
                            tempMat(k,:) = s(i).items;
                        end
                        s = tempMat;
                        break;
                    end
                    
                    s = s.items;
                    % stop iterating because there are no more fields left                   
                    break;                    
                elseif isfield(s(i).(fn{j}),'layerWasSavedWithoutConfig')
                    % handle custom layers saved without config
                    if isfield(s(i),'name')
                        s(i).(fn{j}).name = s(i).name; 
                    elseif isfield(s(i),'class_name') && ~isempty(lastSeqModelName)
                         % layerspec is nested inside a sequential model node                            
                        seqModelNodeIdx = iTGraph.LayerSpecToNodeIdx(lastSeqModelName);
                        childNodeIds = returnUniqueChildNodeIds(iTGraph.NodeStruct{seqModelNodeIdx});
                        layerNodeId = childNodeIds(i);
                        lmd = jsondecode(iTGraph.NodeStruct{layerNodeId}.user_object.metadata);
                        if isfield(lmd,'name')
                           s(i).(fn{j}).name = lmd.name;
                        else
                           % instance name for the layer can not be found,
                           % cannot continue translating the model in this case                            
                           throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:LayerNameNotFoundInConfig')));                        
                        end
                    else
                        % instance name for the layer can not be found,
                        % cannot continue translating the model in this case                            
                        throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:LayerNameNotFoundInConfig')));
                    end

                elseif (strcmp(fn{j},'inbound_nodes'))
                    % handle layers with nested multi input configs
                    inbound_nodes = s(i).(fn{j});
                    if ~isempty(inbound_nodes) && iscell(inbound_nodes) && iscell(inbound_nodes{1}{1}) && length(inbound_nodes{1}{1}) > 3 ...
                            && isstruct(inbound_nodes{1}{1}{4})
                        inbn_struct = inbound_nodes{1}{1}{4};
                        inbn_names = fieldnames(inbn_struct);
                        if ~isempty(inbn_names)
                            % Ignore empty structs
                            for l = 1:numel(inbn_names)
                                % Ignore inbound nodes that are not coming
                                % from other layers (are not cells)
                                if iscell(inbn_struct.(inbn_names{l}))
                                    s(i).(fn{j}){1}(end+1) = {inbn_struct.(inbn_names{l})};
                                end
                            end
                            % delete the redundant nested inbound node entries
                            % and transpose for compatibility with
                            % KerasLayerInsideDAGModel
                            s(i).(fn{j}){1}{1}(end,:)=[];
                            s(i).(fn{j}){1} = s(i).(fn{j}){1}';
                        end
                    elseif ~isempty(inbound_nodes) && iscell(inbound_nodes) && ~iscell(inbound_nodes{1}{1})
                        % add a level of nesting to make this compatible
                        % with KerasLayerInsideDAGModel
                        s(i).(fn{j}){1} = {};
                        s(i).(fn{j}){1} = inbound_nodes;
                    end
                    
                elseif isstruct(s(i).(fn{j}))
                    s(i).(fn{j}) = updateModelConfig(s(i).(fn{j}), iTGraph, lastSeqModelName);
                end
            end
         end
    end
    s1 = s;
end

function smnUniqueChildNodeIds = returnUniqueChildNodeIds(seqmodelnode)
    smnUniqueChildNodeIds = [];
    for nodeIdx = 1:numel(seqmodelnode.children)
        if ~ismember(seqmodelnode.children(nodeIdx).node_id, smnUniqueChildNodeIds)
            smnUniqueChildNodeIds(end+1) = seqmodelnode.children(nodeIdx).node_id; %#ok<AGROW> 
        end
    end
end
