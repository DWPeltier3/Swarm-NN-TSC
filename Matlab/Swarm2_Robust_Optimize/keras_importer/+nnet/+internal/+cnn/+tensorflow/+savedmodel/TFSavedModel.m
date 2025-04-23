classdef TFSavedModel < handle 
    % Representation of a SAVEDMODEL object from TensorFlow 2.x.

%   Copyright 2020-2023 The MathWorks, Inc.
    
    properties
        Version
        KerasManager
        ImportManager
        GraphInfo
        GraphDef
        SavedModelPath
        ServingDefaultOuputsStruct = [];
    end
    
    methods
        function obj = TFSavedModel(path, importManager, importNetwork)
            import nnet.internal.cnn.tensorflow.*;
            savedmodel.util.addLeveldbLibPath(); 
            obj.SavedModelPath = fullfile(path);
            obj.ImportManager = importManager;
            smstruct = jsondecode(nnet.internal.cnn.tensorflow.tf2mex('saved_model', fullfile(obj.SavedModelPath, 'saved_model.pb'))); 
            
            
            obj.Version = smstruct.saved_model_schema_version;
            checkSupportedSavedModelVersion(obj);

            if numel(smstruct.meta_graphs) > 1
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:MoreThanOneGraphsInSavedModel', obj.SavedModelPath)));
            elseif numel(smstruct.meta_graphs) < 1
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:NoGraphInSavedModel', obj.SavedModelPath)));
            elseif ~isfield(smstruct.meta_graphs,'object_graph_def')
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:SubclassedModelNotSupported')));
            end  
            
            TFVersion = smstruct.meta_graphs.meta_info_def.tensorflow_version;
            checkSupportedTFVersion(obj, TFVersion);

            smstruct = checkAndAddKerasMetaDataToSavedModel(obj, TFVersion, smstruct);            
            
            obj.KerasManager = savedmodel.TFKerasManager(smstruct.meta_graphs.object_graph_def, obj.SavedModelPath, importManager, importNetwork);
            obj.GraphInfo = savedmodel.TFGraphInfo(smstruct.meta_graphs.meta_info_def);
            obj.GraphDef = savedmodel.TFGraphDef(smstruct.meta_graphs.graph_def);
            
            % For SSD Nets there might be a 'serving_default' signature that maps outputs
            % to output names in the detection dictionary
            if isfield(smstruct.meta_graphs,'signature_def') && ...
                 isfield(smstruct.meta_graphs.signature_def,'serving_default') && ...
                    isfield(smstruct.meta_graphs.signature_def.serving_default,'outputs') 
                obj.ServingDefaultOuputsStruct = smstruct.meta_graphs.signature_def.serving_default.outputs;
            end
        end
    end
    
    methods (Access=private)
        function checkSupportedSavedModelVersion(this)
			import nnet.internal.cnn.keras.util.*;
            if ~strcmp(this.Version, '1')
                this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:UnsupportedSavedModelVersion', MessageArgs={this.Version});                               
            end
        end
        
	    function checkSupportedTFVersion(this, TFVersion)
            import nnet.internal.cnn.keras.*;
            import nnet.internal.cnn.keras.util.*;
            
            OLDEST_SUPPORTED_TF_VERSION = '2.0.3';
            NEWEST_SUPPORTED_TF_VERSION = '2.10.0';
            ModelPath = this.SavedModelPath;
            
            if ver2num(TFVersion) < ver2num(OLDEST_SUPPORTED_TF_VERSION)
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:TensorFlowVersionTooOld', ...
                    ModelPath, TFVersion, OLDEST_SUPPORTED_TF_VERSION)));
            end
            if ver2num(TFVersion) > ver2num(NEWEST_SUPPORTED_TF_VERSION)
                this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:TensorFlowVersionTooNew', MessageArgs={ModelPath, TFVersion, NEWEST_SUPPORTED_TF_VERSION});
            end
        end

        function smstruct = checkAndAddKerasMetaDataToSavedModel(this, TFVersion, smstruct)
            import nnet.internal.cnn.keras.*;
            import nnet.internal.cnn.keras.util.*;
            if ver2num(TFVersion) > ver2num('2.5.0')
                % keras metadata is saved in a seperate file 'keras_metadata.pb'
                % decode the keras metadata
                if isfile([this.SavedModelPath filesep 'keras_metadata.pb'])
                    kmetadatastruct = jsondecode(nnet.internal.cnn.tensorflow.tf2mex('saved_metadata', fullfile(this.SavedModelPath, 'keras_metadata.pb')));                    
                    % add the metadata to the savedmodel struct
                    for i = 1:numel(kmetadatastruct.nodes)
                        nodeId = kmetadatastruct.nodes(i).node_id + 1; % Add 1 to node-Id for 1-based indexing in MATLAB
                        nodeKMetaData = kmetadatastruct.nodes(i).metadata;
                        smstruct.meta_graphs.object_graph_def.nodes{nodeId, 1}.user_object.metadata = nodeKMetaData;
                    end
                else
                    this.ImportManager.createAndProcessImportIssue(MessageID='nnet_cnn_kerasimporter:keras_importer:KerasMetadataNotFound');
                end
            end
        end
    end
end

