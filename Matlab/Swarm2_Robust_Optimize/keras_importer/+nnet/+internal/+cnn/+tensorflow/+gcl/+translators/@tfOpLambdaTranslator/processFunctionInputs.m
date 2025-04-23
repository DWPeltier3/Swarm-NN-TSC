function [inputCode, MATLABArgIdentifierNames] = processFunctionInputs(this, MATLABInputsStruct)
%   Copyright 2023 The MathWorks, Inc.

% Use all the input information of the OpLambda function to 
% generate code for all inputs and return input names

    inputCode = "";
    MATLABArgIdentifierNames = cell(1, numel(MATLABInputsStruct));
    for i = 1:numel(MATLABInputsStruct)
        MATLABArgIdentifierNames{i} = MATLABInputsStruct{i}.inputName;
        inputCode = inputCode + writeOpLambdaInput(this, MATLABInputsStruct{i}, i);        
    end    
end

function code = writeOpLambdaInput(this, inputStruct, inputNum)
    if inputStruct.isConstant
        % Generate code for a constant input of the OpLambda layer
        % Only scalar (rank 0) constants are supported for now
        if strcmp(inputStruct.inputName, 'dtype')
            % data type input should be written as a string instead of a number
            code = string(inputStruct.inputName) + ".value = '" + num2str(inputStruct.inputValue) + "';";
        else
            if ~isempty(inputStruct.inputValue)
                code = string(inputStruct.inputName) + ".value = " + num2str(inputStruct.inputValue) + ";";
            else
                code = string(inputStruct.inputName) + ".value = [];";
            end
        end
        code = code + newline + string(inputStruct.inputName) + "." + this.RANKFIELDNAME + " = 0;" + newline;        
    else
        % Generate code for an input connection of the OpLambda layer
        % Rank is not saved in this case hence use ndims of the input array
        code = this.LAYERREF + ".InputLabels{" + num2str(inputNum) + "} = " + string(inputStruct.inputName) + ".dims;";
        code = code + newline + string(inputStruct.inputName) +"_dims = max(ndims(" + string(inputStruct.inputName) +"), numel("+ string(inputStruct.inputName) +".dims));";
        code = code + newline + string(inputStruct.inputName) +" = iPermuteToReverseTF(" + string(inputStruct.inputName) + ", ndims(" + string(inputStruct.inputName) +"));";
        code = code + newline + string(inputStruct.inputName) + " = struct('value', " + string(inputStruct.inputName) + ", '" + this.RANKFIELDNAME + "', " + string(inputStruct.inputName) +"_dims);" + newline;        
    end
end
