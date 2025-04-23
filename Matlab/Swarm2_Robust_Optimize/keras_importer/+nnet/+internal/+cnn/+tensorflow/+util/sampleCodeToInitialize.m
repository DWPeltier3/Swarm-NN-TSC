function sampleCode = sampleCodeToInitialize(inputNames, inputData, isInit)
    % Generates sample code demonstrating how to create formatted dlarrays and
    % initialize the dlnetwork

    %   Copyright 2022 The MathWorks, Inc            

    sampleCodeCreateDlarrays = repmat("", numel(inputNames), 1); 
    for i=1:numel(inputNames)
        
        if isInit
            [fmt, idx] = nnet.internal.cnn.tensorflow.util.sortToDLTLabel(inputData{1,i}{1,2});
            inputShape = inputData{1,i}{1,1};
            inputShape = inputShape(idx);
            sampleCodeCreateDlarrays(i) = sprintf("dlX%s = dlarray(ones([%s]), '%s');", num2str(i), num2str(inputShape), fmt);
        else
            sampleCodeCreateDlarrays(i) = "dlX" + num2str(i) + " = dlarray(<dataArray" + num2str(i) + ">, """ + inputData{i} + """);";
        end
    end
    dlarrayList = join([repmat("dlX", numel(inputNames), 1), num2str([1:numel(inputNames)]')], "");
    dlarrayList = join(dlarrayList', ', ');
    sampleCodeInitialize = "net = initialize(net, " + dlarrayList + ");";
    sampleCode = join([sampleCodeCreateDlarrays; sampleCodeInitialize], newline);
end