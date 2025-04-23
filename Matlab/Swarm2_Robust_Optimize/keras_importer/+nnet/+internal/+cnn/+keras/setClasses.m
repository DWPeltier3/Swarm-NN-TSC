function internalLayers = setClasses(internalLayers, classes)
% setClasses   Set classes on a cell array of internal layers

% Copyright 2018 The MathWorks, Inc.

for i = 1:numel(internalLayers)
    if isa(internalLayers{i},'nnet.internal.cnn.layer.CrossEntropy')
        if iIsAuto(classes)
            %Provide default class names
            iWarningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:FillingInClasses');
            internalLayers{i}.Categories = 'default';            
        else
            if internalLayers{i}.NumClasses ~= numel(classes)
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ClassesMismatchSize',...
                    internalLayers{i}.NumClasses,numel(classes))));
            end
            internalLayers{i}.Categories = classes;
        end        
    elseif isa(internalLayers{i},'nnet.internal.cnn.layer.MeanSquaredError')
        if ~iIsAuto(classes)
            throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ClassesForRegression')));
        end
    end
end
end

function tf = iIsAuto(val)
tf = isequal(string(val), "auto");
end

function iWarningWithoutBacktrace(msgID)
nnet.internal.cnn.keras.util.warningWithoutBacktrace(msgID)
end