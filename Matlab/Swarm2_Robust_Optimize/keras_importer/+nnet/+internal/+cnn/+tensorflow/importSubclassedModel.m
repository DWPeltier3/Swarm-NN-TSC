function [dlnet] = importSubclassedModel(savedModel)
    %
    %   Copyright 2022-2023 The MathWorks, Inc.
    
    import nnet.internal.cnn.tensorflow.*;
    try
        visitor = savedmodel.ObjectGraphVisitor(savedModel.KerasManager.InternalTrackableGraph, savedModel.GraphDef, savedModel.SavedModelPath, savedModel.ServingDefaultOuputsStruct, savedModel.ImportManager); 
        visitor.traverse();
        dlnet = visitor.TranslatedRootObject.Instance;

        % Try to extract input shape information from the model
        inputShapes = extractInputShapes(visitor.TranslatedRootObject.FunctionDef);
        savedModel.ImportManager.ImportedInputs = inputShapes;

    catch
         throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:SubclassedModelNotSupported')));
    end

    function inputShapes = extractInputShapes(fcn)
        % Extract input shapes from the translated function definition
    
        inputShapes = {};
        if ~isempty(fcn) && isprop(fcn,'attr') && ...
            ~isempty(fcn.attr) && isfield(fcn.attr,'x_input_shapes') && ...
            ~isempty(fcn.attr.x_input_shapes) && isfield(fcn.attr.x_input_shapes,'list') && ...
            ~isempty(fcn.attr.x_input_shapes.list) && isfield(fcn.attr.x_input_shapes.list,'shape') && ...
            ~isempty(fcn.attr.x_input_shapes.list.shape) && any(~cellfun(@isempty,{fcn.attr.x_input_shapes.list.shape.dim}))

            inputIdx = find(~cellfun(@isempty,{fcn.attr.x_input_shapes.list.shape.dim}));
            numInputs = numel(inputIdx);

            inputShapes = cell(1, numInputs);
            
            for i = 1:numInputs
                inputShape = [];
                inputFormat = '';
                for j = 1:numel(fcn.attr.x_input_shapes.list.shape(inputIdx(i)).dim)
                    inputShape(end+1) = str2double(fcn.attr.x_input_shapes.list.shape(inputIdx(i)).dim(j).size); %#ok<AGROW>
                    inputFormat(end+1) = 'U'; %#ok<AGROW>
                end
                inputShapes{i} = {inputShape, inputFormat};
            end
            
        end
    end
end

