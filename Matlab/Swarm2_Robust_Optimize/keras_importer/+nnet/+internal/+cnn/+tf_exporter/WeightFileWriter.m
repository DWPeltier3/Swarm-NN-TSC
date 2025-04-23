classdef WeightFileWriter

    %   Copyright 2022 The MathWorks, Inc.

    properties (SetAccess=private)
        % weightStruct has one field per layer. The field name is the layer
        % name. The value of a field is a 1x3 cell array: {weightNames
        % (string), weightArrays (cell), weightShapes (cell)}.
        weightStruct struct
    end

    methods
        function this = WeightFileWriter(weightStruct)
            this.weightStruct = weightStruct;
        end

        function write(this, filename)
            % h5 File format:
            % /.NumLayers                           % attribute on /
            % (for each layer):
            %     /layerName.Name                   % attribute on /layerName
            %     /layerName.NumVars                % attribute on /layerName
            %     (for each var):
            %         /layerName/varName.WeightNum  % attribute on /layerName/varName
            %         /layerName/varName.Name       % attribute on /layerName/varName
            %         /layerName/varName.Shape      % attribute on /layerName/varName
            %         /layerName/varName            % An h5 dataset
            %
            % Need to write in "prefix" ordering (data before attributes)
            % so the h5 groups exist before we attach attributes to them.
            %
            if exist(filename,'file')
                delete(filename);
            end
            % Loop over layers
            layerNames = fieldnames(this.weightStruct);
            numLayers  = numel(layerNames);
            for i = 1:numLayers
                % Write a layer
                layerName       = layerNames{i};
                layerLoc        = "/" + layerName;
                weightData      = this.weightStruct.(layerNames{i});
                weightNames     = weightData{1};
                weightArrays    = weightData{2};
                weightShapes    = weightData{3};
                numVars         = numel(weightArrays);
                for var = 1:numVars
                    % write a var's data, then attach name and shape
                    varName = weightNames(var);
                    varShape = weightShapes{var};
                    varData = iExtractIfDlarray(weightArrays{var});
                    varLoc = layerLoc + "/" + varName;
                    iWriteH5Array(filename, varLoc, varData, 'single');
                    h5writeatt(filename, varLoc, 'WeightNum', var-1);
                    h5writeatt(filename, varLoc, 'Name', varName);
                    h5writeatt(filename, varLoc, 'Shape', varShape);
                end
                % Attach layer's name and numVars
                h5writeatt(filename, layerLoc, 'Name', layerName);
                h5writeatt(filename, layerLoc, 'NumVars', numVars);
            end
            % Create the h5 file now if there were no weights
            if numLayers == 0
                h5create(filename, "/0" , 1);   % Dummy dataset
            end
            % Attach numLayers to the root
            h5writeatt(filename, "/", 'NumLayers', numLayers);
        end
    end
end

function iWriteH5Array(h5fileName, locationString, A, DataType)
% Write the array as an Nx1 vector of the given data type.
A = cast(A, DataType);
h5create(h5fileName, locationString, numel(A), 'DataType', DataType);
h5write(h5fileName, locationString, A(:));
end

function A = iExtractIfDlarray(A)
if isa(A, 'dlarray')
    A = extractdata(A);
end
end