function [MATLABOutputName, MATLABArgIdentifierNames, numOutputs] = prepareNode(~, node_def, nameMap, numOutputMap, multiOutputNameMap, modelNodeToOutputMap)
    % This function pre-processes the incoming node_def for translation.
    % Primarily, this method legalizes the input and output variable names
    % of the generated code. The method also tracks the number of outputs 
    % of an op. 

%   Copyright 2020-2021 The MathWorks, Inc.

    % count and record the number of outputs of this current op 
    try
        if isfield(node_def.attr, 'DerivedOutputNodes')
            numOutputs = numel(node_def.attr.DerivedOutputNodes);
        elseif isfield(node_def.attr, 'x_output_shapes') 
            numOutputs = numel(node_def.attr.x_output_shapes.list.shape); 
        else 
            numOutputs = 1; 
        end
    catch 
        % TODO: Actually get the num of outs. 
        numOutputs = 1; 
    end 
    numOutputMap(node_def.name) = numOutputs; 

    % Initialize multiOutputNameMap for ops with multiple outputs
    if numOutputs > 1 && ~isKey(multiOutputNameMap, node_def.name)
        multiOutputNameMap(node_def.name) = [];
    end

    % manually convert identifiers if necessary
    MATLABOutputName = nameMap(node_def.name);
    MATLABArgIdentifierNames = {};
    if iscell(node_def.input)
        for i = 1:numel(node_def.input)
            inputParts = strsplit(node_def.input{i}, ":");
            if isKey(nameMap, inputParts{1})
                MATLABArgIdentifierNames{end + 1} = nameMap(inputParts{1});
            elseif inputParts{1}(1) == '^' && isKey(nameMap, inputParts{1}(2:end))
                % Skip these arguments.
            else
                MATLABArgIdentifierNames{end + 1} = node_def.input{i};
            end
            % If this input is a multi-output op, the input is a
            % cell array. So we add that to the re-named variable and index
            % the cell array appropriately
            if isKey(modelNodeToOutputMap, inputParts{1})
                MATLABArgIdentifierNames{i} = [MATLABArgIdentifierNames{i} '{' num2str(modelNodeToOutputMap(inputParts{1})) '}']; 
            elseif isKey(numOutputMap, inputParts{1}) && numOutputMap(inputParts{1}) > 1 
                if numel(inputParts) == 1
                    inputParts{end+1} = '0'; % it is the first output  
                end

                val = multiOutputNameMap(inputParts{1});
                if isempty(val)
                    MATLABArgIdentifierNames{i} = [MATLABArgIdentifierNames{i} '{' num2str(str2double(inputParts{end}) + 1) '}']; 
                    if numel(inputParts) > 2 && ~strcmp(node_def.op,'AssignVariableOp')
                        multiOutputNameMap(inputParts{1}) = [val, {inputParts{2}}];
                    end

                elseif any(contains(val, inputParts{2}))
                    locInMap = find(contains(val, inputParts(2)));
                    MATLABArgIdentifierNames{i} = [MATLABArgIdentifierNames{i} '{' num2str(str2double(inputParts{end}) + locInMap) '}']; 
       
                else
%                     MATLABArgIdentifierNames{i} = [MATLABArgIdentifierNames{i} '{' num2str(str2double(inputParts{end}) + numel(val) + 1) '}'];
                    MATLABArgIdentifierNames{i} = [MATLABArgIdentifierNames{i} '{' num2str(str2double(inputParts{end}) + 1) '}']; 
                    if numel(inputParts) > 2 && ~strcmp(node_def.op,'AssignVariableOp')
                        multiOutputNameMap(inputParts{1}) = [val, {inputParts{2}}];
                    end
                end
            end
        end
    elseif ~isempty(node_def.input)
        inputParts = strsplit(node_def.input, ":");
        if isKey(nameMap, inputParts{1})
            MATLABArgIdentifierNames = nameMap(inputParts{1});
        else
            MATLABArgIdentifierNames = node_def.input{i};
        end
    
        if isKey(modelNodeToOutputMap, inputParts{1})
            MATLABArgIdentifierNames{i} = [MATLABArgIdentifierNames{i} '{' num2str(modelNodeToOutputMap(inputParts{1})) '}']; 
        elseif isKey(numOutputMap, inputParts{1}) && numOutputMap(inputParts{1}) > 1
			% If this input is a multi-output op, the input is a
        	% cell array. So we add that to the re-named variable
            MATLABArgIdentifierNames = MATLABArgIdentifierNames + "{" + num2str(str2double(inputParts{end}) + 1) + "}"; 
        end
    end
end
