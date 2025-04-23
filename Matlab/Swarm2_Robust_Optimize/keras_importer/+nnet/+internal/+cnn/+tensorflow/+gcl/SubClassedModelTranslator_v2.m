classdef SubClassedModelTranslator_v2 < handle
%   Copyright 2022-2024 The MathWorks, Inc.

    properties
        ModelClass 
        FunctionDef % TFGraphDef representing 
        GraphDef
        SavedModelPath 
        InternalTrackableGraph
        ImportManager % Import manager reference
        ConvertedGraph
        TranslatedChildren
        
        LayerMetaData
        LegalNames
        AllParamsNodeIdx
        LayerVariables 
        Constants
        CapturedInputConstants
        HasUnsupportedOp
        OutputNodeToModelName
        Namespace
        OpFunctionsList % A list of all TensorFlow operator functions used in this layer
        PackageName
        ModelPath
        IsTopLevelModel
    end
    
    properties (Access = private)
        % Properties used for generating code
        TranslatedFunctionsNameSet % A set that tracks which functions were generated already
        TranslatedFunctions        % A list of FunctionTranslator objects that have been translated
        queue = {}                 % A queue for pseudo-recursive calls
    end
    
    properties (Access = private)
        TemplateLocation = which('templateSubClassedLayer.txt')
    end
    
    methods
        function obj = SubClassedModelTranslator_v2(SavedModelPath, InternalTrackableGraph, GraphDef, FunctionDef, TranslatedChildren, ModelClass, ObjectNode, Namespace, PackageName, ModelPath, ImportManager)
            import nnet.internal.cnn.tensorflow.*;
            % Constructor will extract the necessary information from the
            % saved model object for import. 
            obj.ModelClass = ModelClass; 
            obj.GraphDef = GraphDef; 
            obj.FunctionDef = FunctionDef; 
            obj.InternalTrackableGraph = InternalTrackableGraph; 
            obj.SavedModelPath = SavedModelPath; 
            obj.ConvertedGraph = FunctionDef.buildFunctionGraph();
            obj.TranslatedChildren = TranslatedChildren; 
            obj.PackageName = PackageName;
            obj.ModelPath = ModelPath;
            obj.ImportManager = ImportManager;
            
            
            [obj.AllParamsNodeIdx, obj.LayerVariables, obj.CapturedInputConstants] = gcl.getLayerVariables_v2(...
                    SavedModelPath, ...
                    InternalTrackableGraph, ...
                    ObjectNode.children, ...
                    FunctionDef,...
                    obj.ImportManager...
                ); 

            obj.LayerMetaData = containers.Map; 
            obj.HasUnsupportedOp = false; 
            obj.TranslatedFunctionsNameSet = containers.Map(); 
            obj.TranslatedFunctions = []; 
            obj.OutputNodeToModelName = containers.Map(); 
            obj.Namespace = Namespace; 
        end
        
        
        function model = instantiateSubClassedModelWrapper(this, translatedChildren, layerServingDefaultOutputs)
            rehash; 
            modelConstructor = str2func([this.PackageName '.' this.ModelClass]);

            if ~isempty(layerServingDefaultOutputs)
                outputNames = getOutputNamesForServingDefault(layerServingDefaultOutputs);
                model = modelConstructor(this.ModelClass,'SubclassedModel', outputNames);
            else
                model = modelConstructor(this.ModelClass,'SubclassedModel');
            end
            
            for modelName = this.TranslatedChildren.keys
                try
                    if isKey(this.FunctionDef.tfNameToMATLABName, modelName{1})
                        MATLABModelName = this.FunctionDef.tfNameToMATLABName(modelName{1}); 
                    else
                        % MATLABModelName = modelName{1};
                        continue
                    end
                catch
                    continue
                end
                if isa(translatedChildren(modelName{1}).Instance, 'nnet.cnn.LayerGraph')
                    model.(MATLABModelName) = nnet.internal.cnn.tensorflow.gcl.translateTFKerasLayersByName_v2(this.InternalTrackableGraph, this.GraphDef, translatedChildren, this.SavedModelPath, this.ImportManager); 
                    try 
                        model.(MATLABModelName) = dlnetwork(model.(MATLABModelName));  
                    catch
                        this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:ModelInstantiationFailed');                                              
                    end
                else
                    model.(MATLABModelName) = translatedChildren(modelName{1}).Instance; 
                end
            end
            
            % Add learnable parameters to custom layer.
            for curPropertyIdx = 1:numel(this.LayerVariables)
                curVal = this.LayerVariables{curPropertyIdx}.Value;
                model.(this.LegalNames(curPropertyIdx)) = dlarray(single(curVal));
            end
            
            % Add constants to custom layer. 
            graphConstantNames = this.Constants.getConstNames; 
            for curConstIdx = 1:numel(graphConstantNames)
                model.(graphConstantNames{curConstIdx}) = this.Constants.lookupConst(graphConstantNames{curConstIdx});
            end
            
            model = dlnetwork(model, 'Initialize', false); 
        end
        
        
        function writeSubClassedModelWrapper(this)
            constants = nnet.internal.cnn.tensorflow.gcl.TFConstants(this.FunctionDef.tfNameToMATLABName); 
            this.Constants = constants; 
            this.translateForwardPass(this.FunctionDef)
            this.populateOpFunctionsList();
            forwardpass = this.writeForwardPass();             
            
            % Get Template
            [fid, msg] = fopen(this.TemplateLocation, 'rt'); 
            assert(fid ~= -1, ['Custom layer template could not be opened: ' msg '. Make sure the support package has been correctly installed.']);
            code = string(fread(fid, inf, 'uint8=>char')'); 
            fclose(fid);
            
            ModelNames = this.TranslatedChildren.keys; 
            legalNetworkNames = {}; 
            for i = 1:numel(ModelNames)
                try 
                    legalNetworkNames{end+1} = this.FunctionDef.tfNameToMATLABName(ModelNames{i}); 
                catch 
                    this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnableToCreateLegalModelName', MessageArgs={ModelNames{i}});                    
                end 
            end
            legalNetworkNames = strjoin(legalNetworkNames, newline); 
            
            % Variables 
            layerVariableNames = cellfun(@(x)(string(x.curVarName)), this.LayerVariables);
            
            if ~isempty(layerVariableNames)
                legalNames = nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames(layerVariableNames);
            else 
                legalNames = []; 
            end

            nonlearnableParamsCode = string.empty;
            learnableParamsCode = string.empty; 

            for i = 1:numel(this.LayerVariables) 
                if this.LayerVariables{i}.IsLearnable
                    learnableParamsCode(end + 1) = legalNames(i); 
                else 
                    nonlearnableParamsCode(end + 1) = legalNames(i); 
                end
                this.LayerVariables{i}.legalName = legalNames(i); 
            end

            this.LegalNames = legalNames;

            % Extract Arguments
            [highestOutputIdx, inputArgsCode, layerInputs] = this.getArgsCode();
            [extractforwardcall, outputArgsCode] = this.getForwardCallExtractionCode(highestOutputIdx);

            % Model Inputs Code
            [capturedConstCode, layerInputs] = this.writeCapturedInputCode(layerInputs);

            if ~isempty(layerInputs)
                layerInputsCode = strjoin(layerInputs, ", ");
                layerInputsCode = strcat(", ", layerInputsCode);
            else
                layerInputsCode = "";
            end

            % Predict Call
            call = this.writePredictCall(layerInputs, capturedConstCode, inputArgsCode, outputArgsCode);
            
            % Checks for operators that do not support acceleration
            unAcceleratableOps = ["tfCast","tfFloorMod",...
                "tfNonMaxSuppressionV5","tfRandomStandardNormal",...
                "tfSqueeze","tfStatelessIf","tfTopKV2","tfWhere"];
            if isempty(this.OpFunctionsList)
                unAcceleratable = false;
            else
                unAcceleratable = contains(this.OpFunctionsList, unAcceleratableOps);
            end
            
            % Assemble final code string
            if ~any(unAcceleratable)
                inheritance = "& nnet.layer.Acceleratable";
                code = strrep(code, "{{acceleration}}", strjoin(inheritance, newline));
            else
                code = strrep(code, "{{acceleration}}", strjoin("", newline));
            end

            code = strrep(code, "{{autogen_timestamp}}", strjoin(string(datetime('now')), newline));
            code = strrep(code, "{{modelclass}}", this.ModelClass);
            code = strrep(code, "{{networks}}", legalNetworkNames); 
            code = strrep(code, "{{nonlearnables}}", strjoin(nonlearnableParamsCode, newline));
            code = strrep(code, "{{learnables}}", strjoin(learnableParamsCode, newline));
            code = strrep(code, "{{literals}}", strjoin(constants.getConstNames(), newline));
            code = strrep(code, "{{numoutputs}}", num2str(highestOutputIdx));
            code = strrep(code, "{{numinputs}}", num2str(numel(layerInputs)));
            code = strrep(code, "{{modelinputs}}", layerInputsCode);
            code = strrep(code, "{{forwardcall}}", call);
            code = strrep(code, "{{extractforwardcall}}", strtrim(extractforwardcall));
            code = strrep(code, "{{forwardpassdefinition}}", forwardpass);

            code = nnet.internal.cnn.tensorflow.util.indentcode(char(code)); 
            
            % Write generated code to disk
            [fid,msg] = fopen(fullfile(this.ModelPath, this.ModelClass + ".m"), 'w');
            if fid == -1
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnableToCreateCustomLayerFile',this.ModelClass,msg)));
            end
            fwrite(fid, code);
            fclose(fid);
        end

        function objectLoaderResult = correctInputAndOutputArgs(~, objectLoaderResult, layername)
            childsGraph = objectLoaderResult.CodegeneratorObject.ConvertedGraph;
            childsGraphEdges = childsGraph.Edges;
            outgoingEdges = objectLoaderResult.LayerOutputEdges; 
            [numEdges, ~] = size(outgoingEdges);
            if all(endsWith(outgoingEdges(:,1),'PartitionedCall'))
                outgoingEdges = [];
            elseif all(endsWith(outgoingEdges(:,2),'PartitionedCall'))
                outgoingEdges = [];       
            elseif numEdges == 1
                outgoingEdges = [];       
            end

            incomingEdges = objectLoaderResult.LayerInputEdges;
            [numEdges, ~] = size(incomingEdges);
            if all(endsWith(incomingEdges(:,2),'PartitionedCall'))
                incomingEdges = [];
            elseif all(endsWith(incomingEdges(:,1),'PartitionedCall'))
                incomingEdges = [];   
            elseif numEdges > 1
                incomingEdges = [];       
            end

            unusedOutgoingEdgeIndices = [];
            unusedIncomingEdgeIndices = [];
            
            if ~isempty(outgoingEdges)
                returnNodes = struct2table(objectLoaderResult.FunctionDef.ret);
                outputArgNames = {objectLoaderResult.FunctionDef.Signature.output_arg.name};
                tfNameToOutputArgIdxMap = containers.Map();
                IdentityNodeNames = {};
                for i = 1:numel(outputArgNames)
                    nodeParts = strsplit(returnNodes.(outputArgNames{i}),':');
                    IdentityNodeNames{end+1} = nodeParts{1};
                    tfNameToOutputArgIdxMap(nodeParts{1}) = i;
                end
                usedOutgoingEdgeIndices = ismember(childsGraphEdges.EndNodes(:,2), outgoingEdges(:,2));
                allOutgoingEdgeIndices = ismember(childsGraphEdges.EndNodes(:,2), IdentityNodeNames');
                unusedOutgoingEdgeIndices = logical(allOutgoingEdgeIndices -  usedOutgoingEdgeIndices);
            end
            
            if ~isempty(incomingEdges)
                inputNodes = struct2table(objectLoaderResult.FunctionDef.Signature.input_arg);
                inputNodeNames = inputNodes(~ismember(inputNodes.type, {'DT_RESOURCE', 'MATLABONLY'}),1);
                allInputEdges = ismember(childsGraphEdges.EndNodes(:,1),table2cell(inputNodeNames));
                usedInputEdges = ismember(strcat([layername '/'], childsGraphEdges.EndNodes(:,2)),incomingEdges(:,2));
                if ~any(usedInputEdges)
                    incomingEdgeNodes = incomingEdges(:,2);
                    for i = 1 : numel(incomingEdgeNodes)
                        incomingEdgeNodeParts = strsplit(incomingEdgeNodes{i}, "/");
                        incomingEdgeNodes{i} = incomingEdgeNodeParts{2};                        
                    end
                    usedInputEdges = ismember(childsGraphEdges.EndNodes(:,2),incomingEdgeNodes);
                end
                usedInputEdges = allInputEdges & usedInputEdges;
                unusedIncomingEdgeIndices = logical(allInputEdges - usedInputEdges);
            end

            if ~isempty(unusedOutgoingEdgeIndices) || ~isempty(unusedIncomingEdgeIndices) 
                allUnusedEdges = unusedOutgoingEdgeIndices | unusedIncomingEdgeIndices;

                % remove unsed input and output edges
                childsGraph = rmedge(childsGraph,find(allUnusedEdges));                
                indeg = indegree(childsGraph);
                outdeg = outdegree(childsGraph);
                inAndOutDeg = indeg + outdeg;
                inAndOutDegZero = find(inAndOutDeg == 0);
                nodesToRemove = [];
                for j = inAndOutDegZero'
                    if ~strcmp(childsGraph.Nodes.Name(j),'obj')
                        nodesToRemove(end+1) = j; %#ok<*AGROW> 
                    end
                end

                % remove any unconnected nodes
                nodeNamesToRemove = table2cell(childsGraph.Nodes(nodesToRemove,:));
                childsGraph = rmnode(childsGraph, nodesToRemove);

                % remove unused inputs and outputs from child's functiondef
                inputArgIdxToRemove = find(ismember(inputNodes.name,nodeNamesToRemove));
                outputArgIdxToRemove = [];
                for i = 1:numel(nodeNamesToRemove)
                    if isKey(tfNameToOutputArgIdxMap,nodeNamesToRemove{i})                        
                        outputArgIdxToRemove(end+1) = tfNameToOutputArgIdxMap(nodeNamesToRemove{i});
                    end
                end
                objectLoaderResult.FunctionDef.Signature.input_arg(inputArgIdxToRemove) = [];
                
                % remove unused outputs, if they are more than 1
                if numel(objectLoaderResult.FunctionDef.Signature.output_arg) > 1
                    objectLoaderResult.FunctionDef.Signature.output_arg(outputArgIdxToRemove) = [];
                    newOutputArgNames = {objectLoaderResult.FunctionDef.Signature.output_arg.name};
                    retNames = fieldnames(objectLoaderResult.FunctionDef.ret);
                    retNamesToRemoveidx = ~ismember(retNames, newOutputArgNames);
                    retNamesToRemove = retNames(retNamesToRemoveidx);
                    objectLoaderResult.FunctionDef.ret= rmfield(objectLoaderResult.FunctionDef.ret, retNamesToRemove);
                    objectLoaderResult.CodegeneratorObject.ConvertedGraph = childsGraph;
                end
            end
        end

        function computeUsedInputsOutputsForChildren(this, objectLoaderResult)
            % 
            modelNames = this.TranslatedChildren.keys;
            for modelName = modelNames 
                curChild = this.TranslatedChildren(modelName{1}); 
                if  ~isempty(curChild) && strcmp(curChild.APIType, 'SubClassedModel') && isKey(objectLoaderResult.Children,modelName{1})
                    [layerInputEdges,  layerOutputEdges]= this.computeUsedInputsOutputsForChild(modelName{1});
                    layerObjectLoader = objectLoaderResult.Children(modelName{1});
                    layerObjectLoader.LayerInputEdges = layerInputEdges;
                    layerObjectLoader.LayerOutputEdges = layerOutputEdges;
                    layerObjectLoader = this.correctInputAndOutputArgs(layerObjectLoader, modelName{1});
                    objectLoaderResult.Children(modelName{1}) = layerObjectLoader;
                end
            end
        end

        function [inputEdges, outputEdges] = computeUsedInputsOutputsForChild(this, layerName)
            convertedGraphEdges = this.ConvertedGraph.Edges{:,1};
            outputEdges = convertedGraphEdges(startsWith(convertedGraphEdges(:,1), [layerName '/']),:);
            outputEdges = outputEdges(~startsWith(outputEdges(:,2), [layerName '/']),:);
            
            ResourceInputs = {this.FunctionDef.Signature.input_arg.name}; 
            ResourceInputs = ResourceInputs(strcmp({this.FunctionDef.Signature.input_arg.type}, 'DT_RESOURCE'));
            inputEdges = convertedGraphEdges(~startsWith(convertedGraphEdges(:,1), [layerName '/']),:);
            inputEdges = inputEdges(startsWith(inputEdges(:,2), [layerName '/']),:);
            inputEdges = inputEdges(~ismember(inputEdges(:,1), ResourceInputs),:);
        end
        
        function contractAllModels(this)
            % This method will take the subclassed model computational
            % graph and contract all keras models into a single node call.
            % This representation is stored in the ConvertedGraph digraph
            modelNames = this.TranslatedChildren.keys;
            for modelName = modelNames 
                curChild = this.TranslatedChildren(modelName{1}); 
                if true || ~isempty(curChild) %|| strcmp(curChild.APIType, 'SubClassedModel') 
                    this.contractNodesFromLayer(modelName{1});
                end
            end
        end
        
        function convertGraphDef(this)
            % After contraction, this method will re-generate the GraphDef
            % for codegen based on the current ConvertedGraph. 
            % This should be called after contractAllModels. 
            sorted = this.ConvertedGraph.toposort('Order', 'stable');
            
            %this.ConvertedGraph = this.ConvertedGraph.reordernodes(sorted); 
            nameToNodeDefIdx = containers.Map(); 
            for i = 1:numel(this.FunctionDef.node_def)
                nameToNodeDefIdx(this.FunctionDef.node_def(i).name) = i; 
            end
            
            NewNodeDef = cell(size(sorted)); 
            NotANode = []; 
            subModelNameToNewNodeDefIdx = containers.Map(); 
            
            ResourceInputs = {this.FunctionDef.Signature.input_arg.name}; 
            ResourceInputs = ResourceInputs(strcmp({this.FunctionDef.Signature.input_arg.type}, 'DT_RESOURCE')); 

            for i = 1:numel(sorted)
                curName = this.ConvertedGraph.Nodes.Name{sorted(i)}; 
                if isKey(nameToNodeDefIdx, curName) 
                    % check if inputs were contracted nodes. 
                    NewNodeDef{i} = this.FunctionDef.node_def(nameToNodeDefIdx(curName)); 
                    NewNodeDef{i}.input = this.FunctionDef.node_def(nameToNodeDefIdx(curName)).input;
                    if ~isempty(NewNodeDef{i}.input)
                        NewNodeDef{i}.input(startsWith(NewNodeDef{i}.input, '^')) = [];
                    end
                    prev = this.ConvertedGraph.inedges(curName);
                    nodeInputs = NewNodeDef{i}.input;
                    for curInputIdx = 1:numel(nodeInputs)
                        curInputs = this.ConvertedGraph.Edges(prev, :); 
                        foundInput = curInputs(startsWith(curInputs.EndNodes(:,1), nodeInputs{curInputIdx}),:);
                        if isempty(foundInput)
                            nodeInputParts = strsplit(nodeInputs{curInputIdx}, ":");
                            nodeInput = nodeInputParts{1};
                            foundInput = curInputs(startsWith(curInputs.EndNodes(:,1), nodeInput),:);
                            if isempty(foundInput)
                                nodeInputParts = strsplit(nodeInputs{curInputIdx}, "/");
                                nodeInput = nodeInputParts{1};
                                foundInput = curInputs(startsWith(curInputs.EndNodes(:,1), nodeInput),:);
                                if isempty(foundInput)
                                    this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnableToFindNodeInput', MessageArgs={nodeInputs{curInputIdx}, curName});
                                    NewNodeDef{i}.input{curInputIdx} = [nodeInputs{curInputIdx}];
                                    continue;
                                end
                            end
                        end
                        curInput = foundInput;
                        
                        % Cache which output this corresponded to
                        outputNum = num2str(curInput.Weight); 
                        curInput = curInput.EndNodes{1, 1}; 
                        originalInput = NewNodeDef{i}.input{curInputIdx};
                        originalInputParts = strsplit(originalInput, ":");
                        originalInputSubParts = strsplit(originalInputParts{1},'/');
                        outputNodeToModelNameKeys = this.OutputNodeToModelName.keys';
                        if isKey(this.TranslatedChildren, curInput) && (strcmp(this.TranslatedChildren(curInput).APIType,'Functional') || strcmp(this.TranslatedChildren(curInput).APIType,'Sequential'))
                            % functional or sequential submodel
                            if isKey(this.OutputNodeToModelName, originalInputParts{1})
                                % input would depend on the matched 
                                % output number of the functional/sequential model 
                                curInputNodeDef = NewNodeDef{subModelNameToNewNodeDefIdx(curInput)};
                                curInputsOutputs = curInputNodeDef.attr.DerivedOutputNodes;
                                outputNum = num2str(find(strcmp(curInputsOutputs, originalInput)) - 1);
                                NewNodeDef{i}.input{curInputIdx} = [curInput ':' outputNum];

                            elseif numel(originalInputSubParts) > 1
                                if any(startsWith(outputNodeToModelNameKeys, originalInputSubParts{2}))
                                    curInputNodeDef = NewNodeDef{subModelNameToNewNodeDefIdx(curInput)};
                                    curInputsOutputs = curInputNodeDef.attr.DerivedOutputNodes;
                                    outputNum = num2str(find(startsWith(curInputsOutputs, originalInputSubParts{2})) - 1);
                                    NewNodeDef{i}.input{curInputIdx} = [curInput ':' outputNum];
                                else
                                    this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnableToMatchInputs', MessageArgs={curName});
                                    NewNodeDef{i}.input{curInputIdx} = [curInput ':' outputNum];
                                end
                            else
                                this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnableToMatchInputs', MessageArgs={curName});
                                NewNodeDef{i}.input{curInputIdx} = [curInput ':' outputNum];
                            end
                        elseif isKey(this.TranslatedChildren, curInput) && (strcmp(this.TranslatedChildren(curInput).APIType,'SubClassedModel'))
                            % subclassed submodel
                            if isKey(this.OutputNodeToModelName, originalInputParts{1})
                                NewNodeDef{i}.input{curInputIdx} = [curInput ':' outputNum];
                            elseif isKey(this.OutputNodeToModelName, strrep(originalInputParts{1},[curInput '/'],""))
                                curInputNodeDef = NewNodeDef{subModelNameToNewNodeDefIdx(curInput)};
                                curInputsOutputs = curInputNodeDef.attr.DerivedOutputNodes;
                                outputNum = num2str(find(startsWith(curInputsOutputs, strrep(originalInputParts{1},[curInput '/'],""))) - 1);
                                NewNodeDef{i}.input{curInputIdx} = [curInput ':' outputNum];
                            elseif numel(originalInputSubParts) > 1
                                 if any(startsWith(outputNodeToModelNameKeys, originalInputSubParts{2}))
                                    curInputNodeDef = NewNodeDef{subModelNameToNewNodeDefIdx(curInput)};
                                    curInputsOutputs = curInputNodeDef.attr.DerivedOutputNodes;
                                    outputNum = num2str(find(startsWith(curInputsOutputs, originalInputSubParts{2})) - 1);
                                    NewNodeDef{i}.input{curInputIdx} = [curInput ':' outputNum];
                                else
                                    this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnableToMatchInputs', MessageArgs={curName});
                                    NewNodeDef{i}.input{curInputIdx} = [curInput ':' outputNum];
                                end
                            else
                                this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnableToFindNodeInputComingFromSubmodel', MessageArgs={curInput, curName});
                                NewNodeDef{i}.input{curInputIdx} = [curInput ':' outputNum];
                            end
                        end
                    end
                elseif isKey(this.FunctionDef.tfNameToMATLABName, curName) 
                    NotANode(end + 1) = i; 
                elseif isKey(this.LayerMetaData, curName)
                    curNodeInputs = this.ConvertedGraph.Edges(this.ConvertedGraph.inedges(curName), :); 

                    % Delete inputs that are resource names. These no
                    % longer need to be used. 
                    curNodeInputs(ismember(curNodeInputs.EndNodes(:, 1), ResourceInputs), :) = []; 
                    outputNumber = curNodeInputs.Weight;
                    inputNodeIdx = findNodeIdxWithName(NewNodeDef, string(curNodeInputs.EndNodes(:, 1)), i);
                    if ~isempty(inputNodeIdx)
                        inputNodeDef = NewNodeDef{inputNodeIdx};
                        if isfield(inputNodeDef.attr,'DerivedOutputNodes')
                            for j = 1:numel(inputNodeDef.attr.DerivedOutputNodes)
                                outputParts = strsplit(inputNodeDef.attr.DerivedOutputNodes{j},":");
                                for k = 1:numel(this.LayerMetaData(curName).input_edges)
                                    inputEdge = this.LayerMetaData(curName).input_edges{k};
                                    inputEdgeParts = strsplit(inputEdge,'/');
                                    if ismember(outputParts{1},inputEdgeParts)
                                        outputNumber = j-1;
                                        break;
                                    end
                                end
                                if outputNumber ~= curNodeInputs.Weight
                                    break;
                                end
                            end
                        end
                    end

                    formedInputs = string(curNodeInputs.EndNodes(:, 1)) + ":" + string(num2str(outputNumber)); 
                    formedInputs = cellstr(formedInputs); 

                    NewNodeDef{i} = makeNodeDef(curName, 'KerasModelOrLayer', this.TranslatedChildren(curName).APIType, formedInputs);
                    layerOutputNodes = this.LayerMetaData(curName).output_node_defs;
                    layerOutputNodes = strrep(layerOutputNodes,'/PartitionedCall','');
                    NewNodeDef{i}.attr.DerivedOutputNodes = layerOutputNodes;
                    NewNodeDef{i}.attr.DerivedOutputRanks = this.LayerMetaData(curName).outranks;
                    NewNodeDef{i}.attr.InputEdges = this.LayerMetaData(curName).input_edges;
                    if strcmp(NewNodeDef{i}.ParentFcnName,'Functional') || strcmp(NewNodeDef{i}.ParentFcnName,'Sequential')
                        NewNodeDef{i}.attr.LayerToOutName = this.TranslatedChildren(curName).LayerToOutName;
                    else
                        NewNodeDef{i}.attr.LayerToOutName = [];
                    end
                    subModelNameToNewNodeDefIdx(curName) = i;
                    %for j = 1:numel(layerOutputNodes)
                    %    NewNodeDef{i}.attr.DerivedOutputNodes{j} = this.FunctionDef.node_def(nameToNodeDefIdx([this.Namespace '/' layerOutputNodes{j}]));
                    %end
                end
            end
            NewNodeDef(NotANode) = []; 
            this.FunctionDef.node_def = horzcat(NewNodeDef{:}); 
            % Re-generate MATLAB compatible names, if there are any nodes in the function graph
            if numel(this.FunctionDef.node_def) > 0
                this.FunctionDef.cacheMATLABCompatibleNames(this.Namespace); 
            end
            
            % Clear metadata cache. 
            this.LayerMetaData = containers.Map(); 
        end
    end
    
    methods (Access = private)
        function [capturedConstCode, layerInputs] = writeCapturedInputCode(this, layerInputs)
            capturedConstCode = "";
            numRealInputs = numel(layerInputs) - numel(this.CapturedInputConstants);
            for j = 1: numel(this.CapturedInputConstants)
                MATLABInputName = layerInputs{numRealInputs+j};
                constNodeInGraphDef = this.GraphDef.findNodeByName(this.CapturedInputConstants{j});
                currConstCode = nnet.internal.cnn.tensorflow.gcl.translators.tfOpTranslator().writeCapturedInput(constNodeInGraphDef,MATLABInputName, this.Constants);
                capturedConstCode = capturedConstCode + currConstCode;                
            end
            
            % remove all captured inputs since they are constants now
            for k = numel(this.CapturedInputConstants):-1:1
                layerInputs(numRealInputs+k) = [];
            end
        end

        function [highestOutputIdx, inputArgsCode, layerInputs] = getArgsCode(this)
            inputargs = this.FunctionDef.Signature.input_arg; 
            numInputs = numel(this.FunctionDef.Signature.input_arg) - 1; 
            inputArgsCode = cell(numInputs + 1, 1); 
            layerInputs = {};
            curVar = 1; 
            curInput = 1; 
            for i = 1:numInputs
                if strcmp(inputargs(i).type, 'DT_RESOURCE')
                    curLegalVarName = this.LayerVariables{curVar}.legalName; 
                    inputArgsCode{i} = "obj." + (curLegalVarName);
                    curVar = curVar + 1; 
                else
                    inputArgsCode{i} = inputargs(i).name;
                    layerInputs{curInput} = inputargs(i).name;  
                    curInput = curInput + 1; 
                end
            end
            inputArgsCode{end} = 'obj'; 
            
            highestOutputIdx = numel(this.FunctionDef.Signature.output_arg); 
        end

        function [extractforwardcall, outputArgsCode] = getForwardCallExtractionCode(this, highestOutputIdx)
            outputArgsCode = cell(1, highestOutputIdx); 
            extractforwardcall = "";
            for curOut = 1:highestOutputIdx 
                curOutStr = num2str(curOut); 
                outputArgsCode{curOut} = ['temp{' curOutStr '}'];
                prePermutationStr = '';
                postPermutationStr = '';
                if this.IsTopLevelModel && this.ImportManager.OnlySupportDlnetwork
                    % Only permute output for top level subclassed layer
                    prePermutationStr = "iPermuteToForwardTF(";
                    postPermutationStr = ", " + ['temp{' curOutStr '}.rank'] + ")";
                end

                % Add output labels inferred by the layer-level labeler
                extractforwardcall = extractforwardcall + outputArgsCode{curOut}  + " = " + "addOutputLabel(" ...
                    + [outputArgsCode{curOut} ', ' curOutStr ', obj)'] + ";" + newline;
                
                % Layer output extract call
                extractforwardcall = extractforwardcall + ['varargout{' curOutStr '}'] + " = " + ...
                    prePermutationStr + ['temp{' curOutStr '}.value'] + postPermutationStr + ";" + newline;
            end
        end

        function call = writePredictCall(this, layerInputs, capturedConstCode, inputArgsCode, outputArgsCode)
            call = string(['import ' this.PackageName '.ops.*;']);
            
            for i=1:numel(layerInputs)
                inputRank = num2str(numel(this.FunctionDef.attr.x_input_shapes.list.shape(i).dim));
                if this.IsTopLevelModel && this.ImportManager.OnlySupportDlnetwork
                    % Only permute from forward to reverse TF for top level subclassed layer
                    call = call + newline + nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("iPermuteToReverseTF", layerInputs{i}, {layerInputs{i}, inputRank}) + newline;
                else
                    % Permute from labeled reverse TF to unlabeled reverse
                    % TF by setting the isInternal flag
                    isInternal = "true";
                    call = call + newline + nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall("iPermuteToReverseTF", layerInputs{i}, {layerInputs{i}, inputRank, isInternal}) + newline;
                end
            end
            call = call + nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(this.FunctionDef.Signature.legalname, outputArgsCode, inputArgsCode);
            call = capturedConstCode + newline + call;
        end

        function translateForwardPass(this, TopFcn)
            import nnet.internal.cnn.tensorflow.*;
            topFcnTranslator = gcl.FunctionTranslator(TopFcn, this.Constants, true); 
            topFcnTranslator.translateFunction(this.ImportManager, this.ModelClass)
            this.TranslatedFunctions = topFcnTranslator;
            this.TranslatedFunctionsNameSet(TopFcn.Signature.name) = 1; 
            this.queue = topFcnTranslator.SubFunctions; 
            if ~this.HasUnsupportedOp && topFcnTranslator.HasUnsupportedOp
                this.HasUnsupportedOp = true;
            end            
            % Generate subfunctions (dependencies)
            this.translateSubFunctions(); 
        end
        
        function translateSubFunctions(this)
            % FIFO re-calling to generate function dependencies of
            % previously generated function.
            import nnet.internal.cnn.tensorflow.*;
            while ~isempty(this.queue)
                % pop from queue
                subfcnname = this.queue{1}; 
                this.queue(1) = []; 
                
                % only generate the function if it has not been seen before
                if ~isKey(this.TranslatedFunctionsNameSet, subfcnname)
                    this.TranslatedFunctionsNameSet(subfcnname) = 1; 
                    
                    subfcntranslator = gcl.FunctionTranslator(this.GraphDef.findFunction(subfcnname), this.Constants, false); 
                    subfcntranslator.translateFunction(this.ImportManager, this.ModelClass);
                    this.TranslatedFunctions = [this.TranslatedFunctions subfcntranslator]; 
                    this.queue = [this.queue subfcntranslator.SubFunctions]; 
                    if ~this.HasUnsupportedOp && subfcntranslator.HasUnsupportedOp
                        this.HasUnsupportedOp = true;
                    end 
                end
            end
        end

        function populateOpFunctionsList(this)
            for curFunction = 1:numel(this.TranslatedFunctions)
                curTranslatedFunction = this.TranslatedFunctions(curFunction);
                for i = 1:numel(curTranslatedFunction.NodeTranslations)
                    this.OpFunctionsList = [this.OpFunctionsList curTranslatedFunction.NodeTranslations(i).OpFunctions]; 
                end
            end
        end
        
        function code = writeForwardPass(this)
            % Generate the entire forward pass of this custom layer into a
            % string. 
            code = "";
            opsPackageImportStr = ['import ' this.PackageName '.ops.*;'];
            for curFunction = 1:numel(this.TranslatedFunctions)
                code = code + this.TranslatedFunctions(curFunction).emitFunction_v2(opsPackageImportStr); 
            end
        end

        function contractNodesFromLayer(this, layerName)
            members = find(startsWith(this.ConvertedGraph.Nodes.Name, [layerName '/']));
            sg = this.ConvertedGraph.subgraph(members);
            if sg.numnodes == 0 
                % Submodel was not called in the parent model
                % Remove it from Translated Children
                remove(this.TranslatedChildren, layerName);
                return; 
            end
            fcn = this.TranslatedChildren(layerName).FunctionDef; 
            fcnOuts = gatherFcnDefOutputs(fcn);
            CurMetaData.output_node_defs = fcnOuts; 
            
            if isa(this.TranslatedChildren(layerName).Instance, 'dlnetwork') && ~isempty(this.TranslatedChildren(layerName).Instance)
                CurMetaData.numOuts= numel(fcnOuts); 
                %CurMetaData.numOuts = numel(this.TranslatedChildren(layerName).OutputNames); 
            end
            
            outputNames = struct2cell(fcn.ret); 
            outranks = zeros(numel(fields(fcn.ret)), 1);
            for i = 1:numel(outranks)
                outranks(i) = findNodeRanks(fcn, outputNames{i}); 
            end 
            CurMetaData.outranks = outranks; 

            % Keep track of the output node.
            %CurMetaData.output_node_defs = sg.Nodes.Name(sg.outdegree == 0); 
            IdentityNodeInputs = {};
            IdentityNodes = {};
            for i = 1:numel(this.FunctionDef.node_def)
                if strcmp(this.FunctionDef.node_def(i).op,'Identity')
                    IdentityNodeInputs{end+1} = this.FunctionDef.node_def(i).input{1};  
                    IdentityNodes{end+1} = this.FunctionDef.node_def(i).name;
                end
            end
            
            for j = 1:numel(CurMetaData.output_node_defs)                
                if strcmp(this.TranslatedChildren(layerName).APIType,'Functional') || strcmp(this.TranslatedChildren(layerName).APIType,'Sequential')
                    currentOutputName = strsplit(CurMetaData.output_node_defs{j},"/");
                    k = startsWith(IdentityNodeInputs, [layerName '/' currentOutputName{1}]);
                    l = ~startsWith(IdentityNodeInputs(k), [layerName '/']);
                        if any(l)
                            CurMetaData.output_node_defs{j} = IdentityNodeInputs{l};
                            correctedOutput = strsplit(IdentityNodeInputs{l}, ":"); 
                            correctedOutput = correctedOutput{1};
                            this.OutputNodeToModelName(correctedOutput) = layerName;
                        else
                            this.OutputNodeToModelName(CurMetaData.output_node_defs{j}) = layerName;
                        end
                else
                    this.OutputNodeToModelName(CurMetaData.output_node_defs{j}) = layerName;
                end            
            end
            
            
            tg = this.ConvertedGraph.rmnode(members);
            tg = tg.addnode(layerName); 
            layerNodeNames = this.ConvertedGraph.Nodes.Name(members); 
            layerResourceInputs = {fcn.Signature.input_arg.name};
            layerResourceInputs = layerResourceInputs(strcmp({fcn.Signature.input_arg.type}, 'DT_RESOURCE')); 
            layerResourceInputs = strcat([lower(layerName) '_'], layerResourceInputs);
            CurMetaData.input_edges={};
            for node = sg.Nodes.Name'
                prev = this.ConvertedGraph.inedges(node{1});
                prev = this.ConvertedGraph.Edges(prev, :);
                isLayerInput = ismember(prev.EndNodes(:, 1), layerNodeNames);
                if any(~isLayerInput)
                    tg = tg.addedge(prev.EndNodes(~isLayerInput, 1), layerName, prev.Weight(~isLayerInput));                     
                end
                
                isNonResourceLayerInput = ~ismember(prev.EndNodes(:, 1), layerNodeNames) &...
                                            ~ismember(prev.EndNodes(:, 1), layerResourceInputs) &...
                                                ~endsWith(prev.EndNodes(:, 1), 'resource');
                if any(isNonResourceLayerInput)
                    nonResLayerInputs = prev.EndNodes(isNonResourceLayerInput,:)';
                    for k = 1:numel(nonResLayerInputs)
                        CurMetaData.input_edges{end+1}=nonResLayerInputs{k};
                    end
                end

                succ = this.ConvertedGraph.outedges(node{1});
                succ = this.ConvertedGraph.Edges(succ, :);

                isLayerOutput = ismember(succ.EndNodes(:, 2), layerNodeNames); 
                if any(~isLayerOutput)
                    %[~, loc] = ismember(succ.EndNodes(:, 1), CurMetaData.output_node_defs); 

                    tg = tg.addedge(layerName, succ.EndNodes(~isLayerOutput, 2), succ.Weight(~isLayerOutput) ); 
                end
            end
            this.LayerMetaData(layerName) = CurMetaData;
            this.ConvertedGraph = tg; 
        end
    end
end

function mlOutputNames = getOutputNamesForServingDefault(layerServingDefaultOpsStruct)
    mlOutputNames = {};
    tfOutputNames = fieldnames(layerServingDefaultOpsStruct);
    for i = 1:numel(tfOutputNames)
        tfOutputName = tfOutputNames{i};
        outputStruct = layerServingDefaultOpsStruct.(tfOutputName);
        mappedNameParts = strsplit(outputStruct.name,":");
        outputNumber = str2double(mappedNameParts{end}) + 1;
        mlOutputNames{outputNumber} = tfOutputName;
    end
end

function rank = findNodeRanks(fcn, node_name)
    % searches a TFFunction in reverse to find a desired node.
    node_name = strsplit(node_name, ":"); 
    node_name = node_name{1}; 
    node = []; 
    for i = numel(fcn.node_def):-1:1
        if strcmp(fcn.node_def(i).name, node_name)
            node = fcn.node_def(i); 
            break 
        end 
    end 
    if ~isempty(node)
        rank = numel(node.attr.x_output_shapes.list.shape.dim); 
    else
        rank = 0; 
    end 
end 

function networkOuts = gatherFcnDefOutputs(fcn)
% in a functiondef, the outputs are typically assigned after an
% identity operation. However, in nested models, this identity is omitted.
% so we can search for it. 
if isempty(fcn)
    networkOuts = {}; 
    return 
end

networkOuts = cell(numel(fcn.Signature.output_arg), 1); 
for i = 1:numel(fcn.Signature.output_arg)
    cur_output_name = fcn.Signature.output_arg(i).name; 
    cur_output_node = fcn.ret.(cur_output_name);
    cur_output_node = stripOutputNum(cur_output_node);
    networkOuts{i} = findPrev(fcn, cur_output_node);
    
end


    function prev = findPrev(fcn, output_node)
        for j = numel(fcn.node_def):-1:1
            if strcmp(output_node, fcn.node_def(j).name)
                prev = fcn.node_def(j).input; 
                prev = stripOutputNum(prev{1});
                return
            end
        end
        prev = []; 
    end


%fcnGraph = fcn.buildFunctionGraph;

%outs = struct2cell(fcn.ret);
%outs = cellfun(@(x)stripOutputNum(x), outs, 'UniformOutput', false);

%networkOuts = cellfun(@(x)fcnGraph.predecessors(x), outs); 

    function name= stripOutputNum(opName)
        parts = strsplit(opName, ':');
        name = parts{1}; 
    end
end 

function newNodeDef = makeNodeDef(name, op, APIType, input)
    ndStruct.name = name; 
    ndStruct.op = op; 
    ndStruct.input = input; 
    ndStruct.device = []; 
    ndStruct.attr = [];     
    newNodeDef = nnet.internal.cnn.tensorflow.savedmodel.TFNodeDef(ndStruct); 
    newNodeDef.ParentFcnName = APIType;
end

function nodeIdx = findNodeIdxWithName(NodeDefs,nodeName, searchIdx)
    for i = searchIdx:-1:1
        nodeDef = NodeDefs{i};
        if ~isempty(nodeDef) && all(strcmp(nodeDef.name,nodeName))
            nodeIdx = i;
            return
        end
    end
    nodeIdx = [];
end

