classdef (Abstract) LoadTFObjectStrategy 
    % LoadTFObjectStrategy 

    % Instances of this class are responsible for instantiating equivalent
    % DLT models from TensorFlow. A result representation is stored in
    % a ObjectLoaderResult class. 

    properties
        ObjectNode % A struct
        ImportManager % Import manager reference
    end

    methods
        function obj = LoadTFObjectStrategy(ObjectNode, ImportManager)
            obj.ObjectNode = ObjectNode; 
            obj.ImportManager = ImportManager;
        end
    end
    
    methods (Abstract)
        % This method will populate the fields of the ObjectLoaderResult
        % object. The method will expect the Children property of the
        % ObjectLoaderResult to already be completed as we are traversing
        % in a post-order fashion. 
        translateObject(this, ObjectLoaderResult, InternalTrackableGraph, GraphDef, smpath); 
    end

    methods (Access = protected)
        function idx = getChildWithName(this, name)
            % Returns the MATLAB index for a child with a specific local
            % name. Returns empty if not found. 
            idx = []; 
            for i = 1:numel(this.ObjectNode.children)
                if strcmp(this.ObjectNode.children(i).local_name, name)
                    idx = this.ObjectNode.children(i).node_id + 1; 
                    break; 
                end
            end
        end
    end
end

