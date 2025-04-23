function TFFunctionInputs = getFunctionInputs(~, inboundNodes, layerInputNames)
    % This function uses inboundNodes in the TFOpLambda layer Keras Config along 
    % with the inputs of the corresponding placeholder layer to create a cell array 
    % of input variable names and their values for generating the Op's code

    %   Copyright 2023 The MathWorks, Inc.
    
    TFFunctionInputs = {};
    if numel(inboundNodes{1}) > 3 && isstruct(inboundNodes{1}{4})
        nestedStruct = inboundNodes{1}{4};
        inboundNodes{1}(4) = [];
        fns = fieldnames(nestedStruct);
        for i = 1:numel(fns)
            inboundNodes{end+1} = struct(fns{i},nestedStruct.(fns{i})); %#ok<AGROW>            
        end
    end

    inboundConnNum = 1;    
    for i = 1:numel(inboundNodes)
        input = inboundNodes{i};
        if isstruct(input) 
            if numel(input) == 1 && ~isfield(input,'name')
                % Constant input 
                origInputName = fieldnames(input);
                inputName = nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames(origInputName);
                inputName = inputName{1};
                inputVal = input.(origInputName{1});
                isConst = true;
            elseif numel(input) == 3
                % Layer input, not a constant
                if inboundConnNum <= numel(layerInputNames)
                    inputName = layerInputNames{inboundConnNum};
                    inboundConnNum = inboundConnNum + 1;
                else
                    inputName = '_UNK_INPUT';
                end
                inputVal = [];
                isConst = false;
            else
                % 'name' input is never used hence ignore it
                continue;
            end
        elseif iscell(input) && strcmp(input{1}, '_CONSTANT_VALUE')
            inputName = nnet.internal.cnn.tensorflow.gcl.util.iMakeLegalMATLABNames(input(1));
            inputName = inputName{1};
            inputVal = input{end};                            
            isConst = true;
        else
            if inboundConnNum <= numel(layerInputNames)
                inputName = layerInputNames{inboundConnNum};
                inboundConnNum = inboundConnNum + 1;
            else
                inputName = '_UNK_INPUT';
            end
            inputVal = [];
            isConst = false;
        end
        TFFunctionInputs{end+1} = struct('inputName', inputName, 'inputValue', inputVal, 'isConstant', isConst); %#ok<AGROW>        
    end    
end
