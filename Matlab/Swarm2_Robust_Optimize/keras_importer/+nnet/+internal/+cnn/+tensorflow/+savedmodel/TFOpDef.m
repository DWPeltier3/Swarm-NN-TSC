classdef TFOpDef < handle
% TFOpDef Class representation of a TensorFlow function signature. 

%   Copyright 2020-2023 The MathWorks, Inc.
    
    properties
        name 
        input_arg
        output_arg
        control_output
        attr
        summary
        description
        is_commutative
        is_aggregate
        is_stateful
        allows_uninitialized_input
        
        % MATLAB
        MATLABIdentifierName
        legalname % legalized MATLAB name
    end
    
    methods
        function obj = TFOpDef(opdef)
            import nnet.internal.cnn.tensorflow.*;
            obj.name = opdef.name; 
            obj.legalname = gcl.util.iMakeLegalMATLABNames({opdef.name}); 
            obj.legalname = obj.legalname{1}; 
            obj.input_arg = opdef.input_arg; 
            obj.output_arg = opdef.output_arg; 
            obj.control_output = opdef.control_output; 
            obj.attr = opdef.attr; 
            obj.summary = opdef.summary; 
            obj.description = opdef.description; 
            obj.is_commutative = opdef.is_commutative; 
            obj.is_aggregate = opdef.is_aggregate; 
            obj.is_stateful = opdef.is_stateful; 
            obj.allows_uninitialized_input = opdef.allows_uninitialized_input; 
            
            % Add a synthetic input referencing the params object
            if isempty(obj.input_arg)
                obj.input_arg = makeParamsInput(); 
            else 
                obj.input_arg(end + 1) = makeParamsInput(); 
            end
        end
        
        function signaturecode = writeMATLABSignature(this)
            import nnet.internal.cnn.tensorflow.*;
            subroutinename = this.legalname; 
            inputargs = this.input_arg;
            inputargnames = {inputargs.name};
            outputargnames = gcl.util.iMakeLegalMATLABNames({this.output_arg.name}); 
            
            % Split lines after text exceeds some max length. 
            inputargnames = gcl.util.iSplitFcnCalls(inputargnames, numel(subroutinename)); 
            outputargnames = gcl.util.iSplitFcnCalls(outputargnames, 12); 
            
            signaturecode = sprintf("function [%s] = %s(%s)" + newline, strjoin(outputargnames, ", "), ...
                                                              subroutinename, ...
                                                              strjoin(inputargnames, ", "));
        end
        
        function outputassigncode = writeOutputAssignCode(this, ret, nameMap)
            import nnet.internal.cnn.tensorflow.*;
            outputassigncode = ""; 
            outputNames = {this.output_arg.name}; 
            legalOutputNames = gcl.util.iMakeLegalMATLABNames(outputNames); 
            for i = 1:numel(outputNames)
                if numel(outputNames{i}) > namelengthmax
                    outputNames{i} = outputNames{i}(1:namelengthmax); 
                end 
                tfName = ret.(outputNames{i});
                tfName = strsplit(tfName, ':'); 
                matlabName = nameMap(tfName{1});

                if ~strcmp(legalOutputNames{i}, matlabName)
                    % only assign the output if the names differ. 
                    outputassigncode = outputassigncode + legalOutputNames{i} + " = " + matlabName + ";" + newline;    
                end
            end
            
            if outputassigncode ~= ""
                % Only add the extra comment if we needed to assign code. 
                outputassigncode = string(newline) + string(newline) + "% assigning outputs" + newline + outputassigncode; 
            else 
                outputassigncode = string(newline); 
            end 
        end
        
        function convertToMATLABNames(this, nameMap)
            % Makes input arguments unique and matlab compatible. 
            for i = 1:numel(this.input_arg)
                inputParts = strsplit(this.input_arg(i).name, ":"); 
                if isKey(nameMap, inputParts{1})
                    this.input_arg(i).name = nameMap(inputParts{1}); 
                end
            end
        end 
    end
end

function params = makeParamsInput()
% Creates a representation of a function signature input
params = struct; 
% Right now, functions are meant to be called within a predict method of a
% custom layer. This name can change if necessary. 
params.name = 'obj'; 
params.description = ''; 
params.type = 'MATLABONLY'; 
params.type_attr = ''; 
params.number_attr = '';
params.type_list_attr = '';
params.is_ref = 0; 
end

