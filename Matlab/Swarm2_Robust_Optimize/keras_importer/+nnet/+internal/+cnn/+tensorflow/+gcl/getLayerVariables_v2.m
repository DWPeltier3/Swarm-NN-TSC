function [allparamsNodeIdx, layerVariables, capturedInputs] = getLayerVariables_v2(SavedModelPath, InternalTrackableGraph, children, fcn, importManager)
    import nnet.internal.cnn.tensorflow.*;
    import nnet.internal.cnn.keras.util.*;
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
    capturedInputs = {};
    for j = 1:numel(allparamsNodeIdx)
        curLearnableStruct = InternalTrackableGraph.NodeStruct{allparamsNodeIdx(j)}; 
        if ~isfield(curLearnableStruct, 'variable')
            if isfield (curLearnableStruct,'constant')
                capturedInputs{end + 1} = curLearnableStruct.constant.operation; %#ok<AGROW> 
                continue
            else
                importManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnknownInputParamForModule');                
            end
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
