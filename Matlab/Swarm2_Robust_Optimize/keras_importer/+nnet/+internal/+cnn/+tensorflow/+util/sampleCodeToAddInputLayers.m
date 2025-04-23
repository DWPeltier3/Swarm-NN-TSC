function sampleCode = sampleCodeToAddInputLayers(inputNames, inputFormats)
    % Generates sample code demonstrating how to create input layers and add
    % them to the dlnetwork using addInputLayer

    %   Copyright 2022 The MathWorks, Inc            

    sampleCodeCreateLayers = repmat("", numel(inputNames), 1); 
    sampleCodeAddLayers = repmat("", numel(inputNames), 1);
    for i=1:numel(inputNames)
        layerConstructorName = mapFormatToInputLayer(inputFormats(i));
        sampleCodeCreateLayers(i) = "inputLayer" + num2str(i) + " = " + layerConstructorName + "(<inputSize" + num2str(i) + ">, Normalization=""none"");";
        if i==numel(inputNames)
            % The last call to addInputLayer should initialize
            sampleCodeAddLayers(i) = "net = addInputLayer(net, inputLayer" + num2str(i) + ", Initialize=true);";
        else
            sampleCodeAddLayers(i) = "net = addInputLayer(net, inputLayer" + num2str(i) + ");";
        end
    end
    sampleCode = join([sampleCodeCreateLayers; sampleCodeAddLayers], newline);
end

function constructorName = mapFormatToInputLayer(dltInputFormat)
    % Maps dlarray format to corresponding input layer constructor       
    dltInputFormat = string(dltInputFormat);
    switch dltInputFormat
        case "SSCB"
            constructorName = "imageInputLayer";
        case "SSSCB"
            constructorName = "image3dInputLayer";
        case "CB"
            constructorName = "featureInputLayer";
        case {"CBT", "SCBT", "SSCBT", "SSSCBT"}
            constructorName = "sequenceInputLayer";
        otherwise
            % This should not be encountered.
            constructorName = "<inputLayerConstructor>";
    end
end