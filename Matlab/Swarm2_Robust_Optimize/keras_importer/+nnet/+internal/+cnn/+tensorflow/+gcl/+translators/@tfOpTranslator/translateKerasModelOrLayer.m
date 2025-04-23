function result = translateKerasModelOrLayer(this, node_def, MATLABOutputName, MATLABArgIdentifierNames)
% Translate call to a Keras model or layer inside a generated custom layer.
% Data sent to this model/layer must be labeled dlarrays. The data received
% from the model/layer must be converted back to unformatted dlarrays.

%   Copyright 2022-2023 The MathWorks, Inc.

    result = nnet.internal.cnn.tensorflow.gcl.NodeTranslationResult;  
    
    numOutputs = numel(node_def.attr.DerivedOutputRanks); 
    result.NumOutputs = numOutputs; 
    if numOutputs > 1
        outputNames = makeMultipleOutputArgs(this, MATLABOutputName, numOutputs); 
        outputNamesValue = outputNames + ".value"; 
    else
        outputNames = {MATLABOutputName}; 
        outputNamesValue = outputNames + ".value"; 
    end

    for i = 1:numel(MATLABArgIdentifierNames)
        MATLABArgIdentifierNames{i} = ['iAddDataFormatLabels(' MATLABArgIdentifierNames{i} ')']; 
    end 
    
    if strcmp(node_def.ParentFcnName,'Functional')
         MATLABArgIdentifierNames{end+1} = "'Outputs'";
         outputLayerNodes = node_def.attr.DerivedOutputNodes;
         outputLayerNames = '{';
         erasePattern = '|StatefulPartitionedCall' + textBoundary("end");
         for i = 1: numel(outputLayerNodes)
            outputLayerNodeParts = strsplit(outputLayerNodes{i},'/');
            if numel(outputLayerNodeParts) > 1
                if (isKey(node_def.attr.LayerToOutName, outputLayerNodeParts{1}))
                    % append the output name for the output layer
                    outputLayerName = outputLayerNodeParts{1};
                    allOutputs = node_def.attr.LayerToOutName(outputLayerName);
                    outputNumberParts = strsplit(outputLayerNodeParts{end},':');
                    [outputNumber, success] = str2num(outputNumberParts{end});
                    if success    
                        outputNumber = outputNumber + 1;  
                    else
                        outputNumber = 1;
                    end
                    outputLayerName = [outputLayerName '/' allOutputs{outputNumber}]; %#ok<AGROW> 
                elseif isKey(node_def.attr.LayerToOutName, nnet.internal.cnn.keras.makeNNTName(strjoin(outputLayerNodeParts(1:end-1),'/')))
                    outputLayerName = nnet.internal.cnn.keras.makeNNTName(strjoin(outputLayerNodeParts(1:end-1),'/'));
                    allOutputs = node_def.attr.LayerToOutName(outputLayerName);
                    outputNumberParts = strsplit(outputLayerNodeParts{end},':');
                    [outputNumber, success] = str2num(outputNumberParts{end});
                    if success    
                        outputNumber = outputNumber + 1;  
                    else
                        outputNumber = 1;
                    end
                    outputLayerName = [outputLayerName '/' allOutputs{outputNumber}]; %#ok<AGROW>
                elseif isKey(node_def.attr.LayerToOutName, erase(nnet.internal.cnn.keras.makeNNTName(outputLayerNodes{i}),erasePattern))
                    outputLayerName = erase(nnet.internal.cnn.keras.makeNNTName(outputLayerNodes{i}),erasePattern);
                    allOutputs = node_def.attr.LayerToOutName(outputLayerName);
                    outputNumberParts = strsplit(outputLayerNodeParts{end},':');
                    [outputNumber, success] = str2num(outputNumberParts{end});
                    if success    
                        outputNumber = outputNumber + 1;  
                    else
                        outputNumber = 1;
                    end
                    outputLayerName = [outputLayerName '/' allOutputs{outputNumber}]; %#ok<AGROW>      
                end
            else
                outputLayerName = outputLayerNodeParts{end};
                if (isKey(node_def.attr.LayerToOutName, outputLayerName))
                    % append the output name for the output layer
                    allOutputs = node_def.attr.LayerToOutName(outputLayerName);
                    outputNumberParts = strsplit(outputLayerNodeParts{end},':');
                    [outputNumber, success] = str2num(outputNumberParts{end});
                    if success    
                        outputNumber = outputNumber + 1;  
                    else
                        outputNumber = 1;
                    end
                    outputLayerName = [outputLayerName '/' allOutputs{outputNumber}]; %#ok<AGROW> 
                end
            end
            outputLayerNames = [outputLayerNames '''' outputLayerName ''', ' ]; %#ok<AGROW> 
         end
         outputLayerNames(end-1) = '}';
         MATLABArgIdentifierNames{end+1} = outputLayerNames;
    end

    % Generate the code for the predict and forward calls to the Keras
    % model/layer
    predictCode = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(...
       this.LAYERREF + "." + MATLABOutputName + ".predict", outputNamesValue, MATLABArgIdentifierNames) + newline;
    
    forwardCode = nnet.internal.cnn.tensorflow.gcl.util.writeFunctionCall(...
       this.LAYERREF + "." + MATLABOutputName + ".forward", outputNamesValue, MATLABArgIdentifierNames) + newline;
    
    result.Code = "if ~" + this.LAYERREF + ".IsTraining" + newline + predictCode + "else" + newline + forwardCode + "end" + newline; 
    
    for i = 1:numOutputs
        outrank = num2str(node_def.attr.DerivedOutputRanks(i));
        % Convert labeled dlarray output from this model/ layer into
        % unformatted dlarray in reverse TF format
        result.Code = result.Code + outputNamesValue{i} + " = " + "iPermuteToReverseTF(" + outputNamesValue{i} + ", " + outrank + ", true)" + ";" + newline;
        
        % Manually set the output ranks. 
        result.Code = result.Code + outputNames{i} + "." +this.RANKFIELDNAME + " = " + outrank + ";" + newline; 
    end
    
    result.ForwardRank = false;
    result.IsCommenting = true; 
    result.Comment = "Calling dlnetwork"; 
    result.Success = true; 
end