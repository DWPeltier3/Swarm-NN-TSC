function lGraph =  autoSetMinLength(lGraph)
    % autoSetMinLength  Determine MinLength for Sequence Input Layer
    
    % Copyright 2021 The MathWorks, Inc.
    if length(lGraph.InputNames) == 1
        inputLayer          = lGraph.Layers(1);
        minLength           = iGetMinLengthForSequenceInput(lGraph);
        inputLayerReplace   = sequenceInputLayer(inputLayer.InputSize,"Name",inputLayer.Name,"MinLength",minLength);
        lGraph              = replaceLayer(lGraph,inputLayer.Name,inputLayerReplace);
    end
end

function minLength =  iGetMinLengthForSequenceInput(lGraph)
    inputLayer = lGraph.Layers(1);
    minLength = 1;
    while true
        inputLayerReplace = sequenceInputLayer(inputLayer.InputSize,"Name",inputLayer.Name,"MinLength",minLength);
        lGraph = replaceLayer(lGraph,inputLayer.Name,inputLayerReplace);
        try
            [sizes,formats] = deep.internal.sdk.forwardDataAttributes(lGraph);
            minLength = iBinarySearchMinLength(lGraph,ceil(median(minLength/2:minLength)),floor(minLength/2+1),minLength);
            break;
        catch e
            if iCheckSizeError(e.cause)
                minLength = minLength*2;
                continue;
            else
                minLength = iBinarySearchMinLength(lGraph,ceil(median(minLength/2:minLength)),floor(minLength/2+1),minLength);
                break;
            end
        end
    end
end

function minLength =  iBinarySearchMinLength(lGraph,minLength,startValue,endValue)
    if startValue == endValue
        return;
    else
        inputLayer = lGraph.Layers(1);
        inputLayerReplace = sequenceInputLayer(inputLayer.InputSize,"Name",inputLayer.Name,"MinLength",minLength);
        lGraph = replaceLayer(lGraph,inputLayer.Name,inputLayerReplace);
        try
            [sizes,formats] = deep.internal.sdk.forwardDataAttributes(lGraph);
            endValue = minLength;
            minLength = floor(median(startValue:endValue));
            minLength = iBinarySearchMinLength(lGraph,minLength,startValue,endValue);
        catch e
            if iCheckSizeError(e.cause)
                startValue = minLength+1;
                minLength = floor(median(startValue:endValue));
                minLength = iBinarySearchMinLength(lGraph,minLength,startValue,endValue);

            else
                endValue = minLength;
                minLength = floor(median(startValue:endValue));
                minLength = iBinarySearchMinLength(lGraph,minLength,startValue,endValue);
            end
        end
    end
end

function tf = iCheckSizeError(errorCauseList)
    tf = any(cellfun(@(cause)ismember(cause.identifier,{'nnet_cnn:internal:cnn:layer:Convolution1D:FilterSizeLargerThanInput',...
        'nnet_cnn:internal:cnn:layer:MaxPooling1D:PoolSizeLargerThanInput',...
        'nnet_cnn:internal:cnn:layer:AveragePooling1D:PoolSizeLargerThanInput'}),errorCauseList));
end