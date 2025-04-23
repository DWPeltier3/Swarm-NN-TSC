classdef ObjectGraphVisitor < handle
    % A visitor class which traverses an InternalTrackableGraph and
    % translates each object from the bottom up. 

%   Copyright 2022-2023 The MathWorks, Inc.

    properties 
        InternalTrackableGraph % The saved model's InternalTrackableGraph
        GraphDef % The saved models GraphDef
        SavedModelPath % The relative or absolute path of the saved model
        ImportManager % Import manager reference

        TranslatedRootObject % An ObjectLoaderResult that stores the root 
                             % translation + children
        opFunctionsUsed = ""; % List of all Opfunctions used in this model
        hasUnsupportedOp = false;
        servingDefaultOutputs
        packageName
    end

    methods 
        function obj = ObjectGraphVisitor(ITG, GraphDef, SavedModelPath, ServingDefaultOutputs, ImportManager)
            obj.InternalTrackableGraph = ITG; 
            obj.GraphDef = GraphDef; 
            obj.SavedModelPath = SavedModelPath; 
            obj.TranslatedRootObject = []; 
            obj.servingDefaultOutputs = ServingDefaultOutputs;
            obj.ImportManager = ImportManager;
        end

        function generate(this)
            customLayerPath = [pwd filesep '+' this.packageName];
            if ~exist(customLayerPath, 'dir')
                % A submodel could have already created this package containing GCLs
                % Create this package only if it doesn't already exist 
                mkdir(customLayerPath);
            end
            % Root object, by convention is the first nodestruct. 
            rootObjLoaderRes = this.TranslatedRootObject;
            this.TranslatedRootObject = generateRecursive(this, rootObjLoaderRes);
            nnet.internal.cnn.tensorflow.util.writeOpFunctionScripts(this.opFunctionsUsed, customLayerPath, this.hasUnsupportedOp);
            rehash;
        end

        function objLoaderResult = generateRecursive(this, objLoaderResult)
            % Only generate submodels that are connected
            if ~isempty(objLoaderResult.CodegeneratorObject)
                connectedChildObjects = objLoaderResult.CodegeneratorObject.TranslatedChildren.keys';
                if ~isempty(connectedChildObjects)
                    objLoaderResult.Children = objLoaderResult.CodegeneratorObject.TranslatedChildren;
                end    
            end
            childObjectNames = objLoaderResult.Children.keys;
            if ~isempty(childObjectNames)
                for i = 1:numel(childObjectNames)
                    translatedChild = generateRecursive(this, objLoaderResult.Children(childObjectNames{i}));
                    objLoaderResult.Children(childObjectNames{i}) = translatedChild;
                end
            end

            if objLoaderResult.HasFcn 
                strategy = objLoaderResult.TranslationStrategy;
                strategy.translateObject(objLoaderResult, ...
                                            this.InternalTrackableGraph, ...
                                            this.GraphDef, ...
                                            this.SavedModelPath);
                if ~isempty(objLoaderResult.CodegeneratorObject)
                    this.opFunctionsUsed = [this.opFunctionsUsed objLoaderResult.CodegeneratorObject.OpFunctionsList]; 
                    this.hasUnsupportedOp = objLoaderResult.CodegeneratorObject.HasUnsupportedOp;                
                end
            end

        end

        function traverse(this)
            import nnet.internal.cnn.keras.util.*;
            % create package for generated layers
            if isempty(this.ImportManager.PackageName)
                s = what(this.SavedModelPath);
                p = strsplit(s(end).path, filesep);
                smName = p{end};
                this.packageName = smName;
            else
                this.packageName = this.ImportManager.PackageName;
            end
            % Create valid package name
            if ~(isvarname(this.packageName))
                this.packageName = matlab.lang.makeValidName(this.packageName);
                this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnspecifiedPackageName', MessageArgs={smName,this.packageName});                
            end
            % Root object, by convention is the first nodestruct. 
            this.TranslatedRootObject = traverseRecursive(this, 1); 
            this.generate();
        end

        function translatedObj = traverseRecursive(this, mlidx)
            % Post order traverse the object graph. We will translate the
            % model from the bottom-up. 
            Node = this.InternalTrackableGraph.NodeStruct{mlidx};
            if ~isfield(Node, 'user_object')
                translatedObj = [];
                return; 
            end
            % strategy class which handles instantiation of a model object
            switch Node.user_object.identifier
                case {'_generic_user_object', '_tf_keras_model', '_tf_keras_layer'}
                    strategy = nnet.internal.cnn.tensorflow.savedmodel.LoadSubClassedStrategy(Node, this.ImportManager);
                case {'_tf_keras_network', '_tf_keras_sequential'}
                    % Decode Keras metadata if available
                    if ~isempty(Node.user_object.metadata)
                        node_decoded = jsondecode(Node.user_object.metadata);
                        if ismember(node_decoded.class_name, {'Functional', 'Sequential', 'Model'})
                            strategy = nnet.internal.cnn.tensorflow.savedmodel.LoadFunctionalSequentialStrategy(Node, this.ImportManager);
                        else
                            strategy = nnet.internal.cnn.tensorflow.savedmodel.LoadSubClassedStrategy(Node, this.ImportManager);
                        end
                    else
                        strategy = nnet.internal.cnn.tensorflow.savedmodel.LoadSubClassedStrategy(Node, this.ImportManager);
                    end
                otherwise
                    strategy = [];
            end
            
            if isempty(strategy) 
                % Child is not a translatable object. function, map, etc. 
                translatedObj = [];
                return;
            end
            
            translatedObj = nnet.internal.cnn.tensorflow.savedmodel.ObjectLoaderResult;

            % We do not need to traverse further down the tree if the
            % current node is a sequential or functional model.
            if ~isa(strategy, "nnet.internal.cnn.tensorflow.savedmodel.LoadFunctionalSequentialStrategy")
                duplicateChildNames = 0;
                for i = 1:numel(Node.children)
                    % skip keras_api objects 
                    if strcmp(Node.children(i).local_name, 'keras_api')
                        continue;
                    end
                    
                    translatedChild = traverseRecursive(this, Node.children(i).node_id + 1);

                    if ~isempty(translatedChild)
                        child_struct = this.InternalTrackableGraph.NodeStruct{Node.children(i).node_id + 1}; 

                        % Gather full Namespace of this child as its key
                        namespace = getObjectNamespace(this, Node.children(i).node_id + 1);

                        % Decode Keras metadata if available to get name for child
                        if ~isempty(child_struct.user_object.metadata)
                            name = jsondecode(child_struct.user_object.metadata);
                            name = name.name;
                        else
                            name = strsplit(namespace, filesep);
                            name = name{end};
                            name = char(nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames(string(name)));
                        end

                        translatedObj.Namespace = namespace; 
                        if ~isKey(translatedObj.Children, name)
                            translatedObj.Children(name) = translatedChild;
                        else
                            duplicateChildNames = duplicateChildNames + 1;
                            translatedObj.Children([name '_' num2str(duplicateChildNames)]) = translatedChild;
                        end
                    end
                end
            end

            customLayerPath = [pwd filesep '+' this.packageName];
            % Preprocess current Node and return the result
            strategy.preProcessTranslatorObject( ...
                    translatedObj, ...
                    this.InternalTrackableGraph, ...
                    this.GraphDef, ...
                    this.SavedModelPath, ...
                    this.packageName, ...
                    customLayerPath);

            if mlidx == 1
                % Mark the top level subclassed layer in the translated obj
                translatedObj.IsTopLevelLayer = true;
                if ~isempty(this.servingDefaultOutputs)
                    translatedObj.LayerServingDefaultOutputs = this.servingDefaultOutputs;
                end
            end
        end

        function namespace = getObjectNamespace(this, nodeidx)
            root_node_idx = 1; 
            namespace = this.InternalTrackableGraph.DependencyGraph.shortestpath(num2str(root_node_idx), num2str(nodeidx)); 
            for i = 1:numel(namespace) 
                curObj = this.InternalTrackableGraph.NodeStruct{str2double(namespace{i})}.user_object; 
                if isempty(curObj.metadata) 
                    fcn = getTopLevelFunction(this, this.InternalTrackableGraph.NodeStruct{str2double(namespace{i})});
                    if ~isempty(fcn)
                        namespace{i} = fcn.Signature.name; 
                    else
                        namespace{i} = '';
                    end
                else
                    curObj = jsondecode(curObj.metadata); 
                    namespace{i} = curObj.name; 
                end
            end
            
            if numel(namespace) > 1
                namespace = strjoin(namespace(2:end), '/');
            else 
                namespace = ''; 
            end 
        end

        function fcn = getTopLevelFunction(this, nodestruct)
            % Get the TensorFlow graph that points to a layer's 'call' method
            fcn = [];
            children = nodestruct.children; 
            for childidx = 1:numel(children)
                child = children(childidx); 
                if strcmp(child.local_name, 'call_and_return_conditional_losses') || strcmp(child.local_name, 'call_and_return_all_conditional_losses')
                    fcn = this.InternalTrackableGraph.NodeStruct{child.node_id + 1}; 
                    if isempty(fcn.function.concrete_functions)
                        fcn = [];  
                    else 
                        fcn = fcn.function.concrete_functions{1}; 
                    end
                    fcn = this.GraphDef.findFunction(fcn);
                    break;
                end
            end
        end

        function writeOpFunctionScripts(this, customLayerPath)
            % Create the ops subpackage
            p = strsplit(customLayerPath, '+');
            packageName = p{end}; %#ok<PROPLC>
            opsPackage = [customLayerPath filesep '+ops'];
            spkgOpFileLocation = [fileparts(which('nnet.internal.cnn.tensorflow.importTensorflowLayers')) filesep 'op' filesep];
            if ~isfolder(opsPackage)
                mkdir(opsPackage);
            end

            % Copy util functions
            copyfile([spkgOpFileLocation 'sortByLabel.m'],opsPackage,'f');
            copyfile([spkgOpFileLocation 'sortToTFLabel.m'],opsPackage,'f');
			copyfile([spkgOpFileLocation 'addOutputLabel.m'],opsPackage,'f');
            copyfile([spkgOpFileLocation 'iExtractData.m'],opsPackage, 'f');
            copyfile([spkgOpFileLocation 'iAddDataFormatLabels.m'],opsPackage, 'f');
            
            % Write the opfunction scripts needed for the custom layers
            uniqueOpFunctionsUsed = unique(this.opFunctionsUsed);
            uniqueOpFunctionsUsed(cellfun('isempty',uniqueOpFunctionsUsed)) = [];
            for i = 1:numel(uniqueOpFunctionsUsed)
                % Read base op file
                opFileName = strcat(uniqueOpFunctionsUsed(i),".m"); 
                [fid,msg] = fopen(strcat(spkgOpFileLocation,opFileName));
                % If the support package is correctly installed we expect to
                % find the operator function template, throw an assertion failure if
                % that is not the case
                assert(fid ~= -1, strcat('Failed to find operator function template for ',opFileName,' : ',msg));
                code = textscan(fid, '%s', 'Delimiter','\n', 'CollectOutput',true);
                code = code{1};
                fclose(fid);
                
                % add import line and write back
                [fid,msg] = fopen(strcat(opsPackage,filesep,opFileName), 'w');
                if fid == -1
                                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnableToCreateOpFunctionFile',opFileName,msg)));
                end
                for j = 1 : length(code)
                    if (strcmp(code{j},'%{{import_statement}}'))
                        fprintf(fid, '%s\n', ['import ' packageName '.ops.*;']); %#ok<PROPLC>
                    else
                        fprintf(fid, '%s\n', code{j});
                    end
                end            
                fclose(fid);
            end        
        end
    end
end