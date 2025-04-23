classdef TFFunction < handle
% TFFunction Class representation of an entire TensorFlow function.
% Stores the function signature in (Signature) as a TFOpDef and its
% graph definition in node_def. 

%   Copyright 2020-2023 The MathWorks, Inc.
    
    properties
        Signature
        attr
        arg_attr
        resource_arg_unique_id
        node_def
        ret
        control_ret
        
        tfNameToMATLABName
        tfModelNodeToModelOut 
    end
    
    methods
        function obj = TFFunction(FunctionDef)
            import nnet.internal.cnn.tensorflow.*;
            obj.Signature = savedmodel.TFOpDef(FunctionDef.signature); 
            obj.attr = FunctionDef.attr; 
            obj.arg_attr = FunctionDef.arg_attr; 
            obj.resource_arg_unique_id = FunctionDef.resource_arg_unique_id; 
            obj.node_def = savedmodel.TFNodeDef.empty(); 
            for i = 1:numel(FunctionDef.node_def)
                obj.node_def(i) = savedmodel.TFNodeDef(FunctionDef.node_def(i)); 
            end
            obj.ret = FunctionDef.ret; 
            obj.control_ret = FunctionDef.control_ret; 
            
            obj.tfNameToMATLABName = containers.Map; 
            obj.tfModelNodeToModelOut = containers.Map; 
            obj.cacheMATLABCompatibleNames;
        end
                
        function code = writeInputRankTracker(this, rankFieldName)
            import nnet.internal.cnn.tensorflow.*;
            code = "" + newline; 
            for i = 1:numel(this.attr.x_input_shapes.list.shape)
                curInput = this.Signature.input_arg(i); 
                curInputType = curInput.type;
                if ~strcmp(curInputType, 'DT_RESOURCE') && ~strcmp(curInputType, 'MATLABONLY')
                    if this.attr.x_input_shapes.list.shape(i).unknown_rank
                        code = code + gcl.util.writeFunctionCall("ndims", {curInput.name + "_" + rankFieldName}, {curInput.name}) + newline; 
                        rankVal = curInput.name + "_" + rankFieldName;
                    else 
                        inputRank = num2str(numel(this.attr.x_input_shapes.list.shape(i).dim)); 
                        rankVal = inputRank; 
                    end
                    structArgs = strjoin(["'value'", curInput.name, "'" + rankFieldName + "'", string(rankVal)], ", "); 
                    code = code + gcl.util.writeFunctionCall("struct", curInput.name, structArgs) + newline; 
                end 
            end 
        end

        function code = writeInputRankTracker_v2(this, rankFieldName)
            import nnet.internal.cnn.tensorflow.*;
            code = ""; 
            for i = 1:numel(this.attr.x_input_shapes.list.shape)
                curInput = this.Signature.input_arg(i); 
                curInputType = curInput.type;
                if ~strcmp(curInputType, 'DT_RESOURCE') && ~strcmp(curInputType, 'MATLABONLY')
                    if this.attr.x_input_shapes.list.shape(i).unknown_rank
                        code = code + gcl.util.writeFunctionCall("ndims", {curInput.name + "_" + rankFieldName}, {curInput.name}) + newline; 
                        rankVal = curInput.name + "_" + rankFieldName;
                    else 
                        inputRank = num2str(numel(this.attr.x_input_shapes.list.shape(i).dim)); 
                        rankVal = inputRank; 
                    end
                    structArgs = strjoin(["'value'", curInput.name, "'" + rankFieldName + "'", string(rankVal)], ", "); 
                    code = code + gcl.util.writeFunctionCall("struct", curInput.name, structArgs) + newline; 
                end 
            end 
        end
        
        function fcn = buildFunctionGraph(this)
            % Returns the computational graph of a function as a digraph 
            fcn = digraph();
            for i = 1:numel(this.Signature.input_arg)
                fcn = fcn.addnode(this.Signature.input_arg(i).name); 
            end 

            for i = 1:numel(this.node_def)
                nodename = strsplit(this.node_def(i).name, ":");
                nodename = nodename{1}; 
                fcn = fcn.addnode(nodename);
                for j = 1:numel(this.node_def(i).input)
                    if startsWith(this.node_def(i).input(j), '^')
                        % skip the nodes with ^
                        continue; 
                    end 
                    inputnode = strsplit(this.node_def(i).input{j}, ":"); 
                    % if the input has multiple output, store that as a
                    % weight in the digraph. 
                    if numel(inputnode) > 1 
                        inputnodeoutput = str2double(inputnode{end}); 
                    else 
                        inputnodeoutput = 0; 
                    end 
                    inputnode = inputnode{1}; 
                    fcn = fcn.addedge(inputnode, nodename, inputnodeoutput); 
                end
            end
        end


    end
    methods (Access = public)
        function cacheMATLABCompatibleNames(this, namespace)
            % This function takes all of the names in this function's scope
            % and maps the original names into unique MATLAB compatible
            % names for use in the code generation. The lookup is stored in
            % the tfNameToMATLABName map in this class. 
            import nnet.internal.cnn.tensorflow.*;
            this.tfNameToMATLABName = containers.Map(); 
            matlabNames = gcl.util.iMakeLegalMATLABNames([{this.node_def.name} {this.Signature.input_arg.name}]);
            
            for i = 1:numel(this.node_def)                
                this.tfNameToMATLABName(this.node_def(i).name) = matlabNames{i};
                if strcmp(this.node_def(i).op, 'KerasModelOrLayer') 
                    % We need to cache the internal output too
                    for j = 1:numel(this.node_def(i).attr.DerivedOutputNodes)
                        outputNode = this.node_def(i).attr.DerivedOutputNodes{j}; 
                        %this.tfNameToMATLABName(outputNode.name) = [matlabNames{i}]; % num2str(j - 1)]; 
                        if nargin == 2 && ~isempty(namespace)
                            if (strcmp(this.node_def(i).ParentFcnName,'Functional') || strcmp(this.node_def(i).ParentFcnName,'Sequential'))
                                outputParts = strsplit(outputNode, ":");
                                this.tfNameToMATLABName(outputParts{1}) = [matlabNames{i}];
                                if numel(this.node_def(i).attr.DerivedOutputRanks) > 1
                                    this.tfModelNodeToModelOut(outputParts{1}) = j;
                                    this.tfModelNodeToModelOut([matlabNames{i} '/StatefulPartitionedCall']) = j; 
                                end
                                varname = outputNode;
                            else
                                varname = [this.node_def(i).name '/' outputNode];  
                            end
                            %varname = outputNode; 
                        else 
                            varname = outputNode;
                        end 
                        this.tfNameToMATLABName(varname) = [matlabNames{i}];
                        this.tfNameToMATLABName([matlabNames{i} '/StatefulPartitionedCall']) = [matlabNames{i}];
                        if numel(this.node_def(i).attr.DerivedOutputRanks) > 1
                            this.tfModelNodeToModelOut(varname) = j;
                            this.tfModelNodeToModelOut([matlabNames{i} '/StatefulPartitionedCall']) = j; 
                        end
                    end
                    
                end 

            end

            if isempty(i) 
                i = 0; 
            end
            
            for j = (1:numel(this.Signature.input_arg))
                this.tfNameToMATLABName(this.Signature.input_arg(j).name) = matlabNames{j + i};
            end
        end
    end
end
