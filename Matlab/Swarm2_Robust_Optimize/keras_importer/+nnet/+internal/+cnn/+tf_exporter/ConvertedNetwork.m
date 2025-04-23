classdef ConvertedNetwork

    %   Copyright 2022 The MathWorks, Inc.

    properties (SetAccess=private)
        allLayerCode            string
        allNetInputNames        string
        allNetOutputNames       string
        allCustomLayerNames     string
        weightStruct            struct = struct
        packagesNeeded          string
        placeholderLayerNames   string      % String array containing all placeholder layer names
        placeholderLayerCodeCell cell       % Cell of String arrays containing all placeholder layer class definitions. One cell per layer class.
    end

    properties
        WarningMessages         message     % Array of warning messages during export
    end

    methods
        function this = ConvertedNetwork(netInputNames, netOutputNames, initialWarnings)
            this.allNetInputNames = netInputNames;
            this.allNetOutputNames = netOutputNames;
            this.WarningMessages = initialWarnings;
        end

        function this = updateFromConvertedLayer(this, convertedLayer)
            % Updater to ensure the contents are well-formed.
            % code accumulation
            this.allLayerCode         = [this.allLayerCode; convertedLayer.layerCode];
            this.allCustomLayerNames  = unique([this.allCustomLayerNames; convertedLayer.customLayerNames]);
            this.packagesNeeded       = unique([this.packagesNeeded; convertedLayer.packagesNeeded]);
            if ~ismember(convertedLayer.placeholderLayerName, this.placeholderLayerNames)
                this.placeholderLayerNames = [this.placeholderLayerNames; convertedLayer.placeholderLayerName];
                this.placeholderLayerCodeCell{end+1} = convertedLayer.placeholderLayerCode;
            end
            % weight accumulation
            if ~isempty(convertedLayer.weightNames)
                this.weightStruct.(convertedLayer.LayerName) = {convertedLayer.weightNames, convertedLayer.weightArrays, convertedLayer.weightShapes};
            end
            % Maybe rename an input tensor
            this = maybeRenameInputTensor(this, convertedLayer);
            % Update warnings
            this = addWarningMessages(this, convertedLayer.WarningMessages);
        end

        function this = addWarningMessages(this, messageArray)
            this.WarningMessages = [this.WarningMessages, messageArray(:)'];
        end

        function writeTextFiles(this, parentDir, packageName, modelFileName, initFileName, READMEFileName)
            % Make directory
            dirname = fullfile(parentDir, packageName);
            if exist(dirname, 'dir')
                warnstate = warning('off','backtrace');
                C = onCleanup(@()warning(warnstate));
                warning(message('nnet_cnn_kerasimporter:keras_importer:exporterFolderExists', dirname));
            else
                mkdir(dirname);
            end
            % Write the files
            writeModelFile(this, modelFileName, packageName);
            writeCustomLayerTemplateFiles(this, parentDir, packageName);
            writeInitFile(this, initFileName, packageName);
            writeREADMEFile(this, READMEFileName, packageName)
        end

        function writeModelFile(this, modelFileName, packageName)
            fileID = fopen(modelFileName,'w');
            if fileID<=0
                error(message('nnet_cnn_kerasimporter:keras_importer:exporterFileOpen', modelFileName));
            end
            C = onCleanup(@()fclose(fileID));
            standardTFImports = ["import tensorflow as tf"; "from tensorflow import keras"; "from tensorflow.keras import layers"];
            if ismember("tfa", this.packagesNeeded)
                additionalImportsStr = "import tensorflow_addons as tfa";
            else
                additionalImportsStr = string.empty;
            end
            if ~isempty(this.placeholderLayerNames)
                customLayerImports = "from " + packageName + ".customLayers." + this.placeholderLayerNames + " import " + this.placeholderLayerNames;
            else
                customLayerImports = string.empty;
            end
            pyImportCode = [
                standardTFImports
                additionalImportsStr
                customLayerImports
                ""];
            defineModelCode = iGenDefineModelCode(this.allNetInputNames, this.allNetOutputNames, this.allLayerCode);
            customLayerCode = genCustomLayerCode(this);
            allCode         = [iFileCreatedByComment(); pyImportCode; defineModelCode; customLayerCode];
            fprintf(fileID, "%s", join(allCode, newline));
            clear C
        end

        function writeCustomLayerTemplateFiles(this, parentDir, packageName)
            % Write the custom layer template files
            if ~isempty(this.placeholderLayerNames)
                % Make directory for exported custom layers
                dirname = fullfile(parentDir, packageName, "customLayers");
                if ~exist(dirname, 'dir')
                    mkdir(dirname);
                end
            end
            for i=1:numel(this.placeholderLayerNames)
                % Write one definition to one file.
                placeholderFilename = fullfile(parentDir, packageName, "customLayers", this.placeholderLayerNames(i)+".py");
                fileID = fopen(placeholderFilename,'w');
                if fileID<=0
                    error(message('nnet_cnn_kerasimporter:keras_importer:exporterFileOpen', placeholderFilename));
                end
                C = onCleanup(@()fclose(fileID));
                % Preface the code with the "created by" comment
                allCode = [iFileCreatedByComment(); this.placeholderLayerCodeCell{i}];
                fprintf(fileID, "%s", join(allCode, newline));
                clear C
            end
        end

        function writeInitFile(~, initFileName, packageName)
            % Write the __init__ file
            fileID = fopen(initFileName,'w');
            if fileID<=0
                error(message('nnet_cnn_kerasimporter:keras_importer:exporterFileOpen', initFileName));
            end
            C = onCleanup(@()fclose(fileID));
            pyImportCode = ["import " + packageName + ".model"; "import os"; ""];
            loadModelCode = [
                "def load_model(load_weights=True, debug=False):"
                "    m = model.create_model()"
                "    if load_weights:"
                "        loadWeights(m, debug=debug)"
                "    return m"
                ""
                ];
            utilityFcnCode  = iGenUtilityFcnCode();
            allCode         = [iFileCreatedByComment(); pyImportCode; loadModelCode; utilityFcnCode];
            fprintf(fileID, "%s", join(allCode, newline));
            clear C
        end

        function writeREADMEFile(this, READMEFileName, packageName)
            % Write the README file
            fileID = fopen(READMEFileName,'w');
            if fileID<=0
                error(message('nnet_cnn_kerasimporter:keras_importer:exporterFileOpen', READMEFileName));
            end
            C = onCleanup(@()fclose(fileID));
            if isempty(this.WarningMessages)
                issuesText = message('nnet_cnn_kerasimporter:keras_importer:exporterREADMENoIssues').getString;
            else
                warningStrings = arrayfun(@(mess)string(mess.getString), this.WarningMessages);
                issuesText = join(warningStrings, string([newline,newline]));
                issuesText = iWrapText(issuesText);
            end
            readmeCode = [
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMECreatedByLine1').getString
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMECreatedByLine2').getString
                string(datetime)
                ""
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEThisPackageContains').getString
                ""
                ""
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEIssuesHeading').getString
                "--------------------------------------------"
                issuesText
                ""
                ""
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEUsageHeading').getString
                "-----"
                ""
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEToLoadModel1').getString
                ""
                "    import " + packageName
                "    model = " + packageName + ".load_model()"
                ""
                ""
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEToLoadModel2').getString
                ""
                "    import " + packageName
                "    model = " + packageName + ".load_model(load_weights=False)"
                ""
                ""
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEToSavedModel').getString
                ""
                "    model.save(<fileName>)"
                ""
                ""
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEToHDF5').getString
                ""
                "    model.save(<fileName>, save_format='h5')"
                ""
                ""
                message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEPackageFilesHeading').getString
                "-------------"
                ""
                "model.py"
                sprintf('\t') + iWrapText(message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEModelDotPy').getString)
                ""
                "weights.h5"
                sprintf('\t') + iWrapText(message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEWeightsDotH5').getString)
                ""
                "__init__.py"
                sprintf('\t') + iWrapText(message('nnet_cnn_kerasimporter:keras_importer:exporterREADMEInitDotPy').getString)
                ""
                ];
            fprintf(fileID, "%s", join(readmeCode, newline));
            clear C
        end

        function writeWeightFile(this, weightFileName)
            weightFileWriter = nnet.internal.cnn.tf_exporter.WeightFileWriter(this.weightStruct);
            write(weightFileWriter, weightFileName);
        end
    end

    methods (Access=protected)
        function this = maybeRenameInputTensor(this, convertedLayer)
            % Note: This code has asserts to make sure developers didn't
            % make mistakes in setting the RenameNetworkInputTensor
            % property.
            if ~isempty(convertedLayer.RenameNetworkInputTensor)
                assert(numel(convertedLayer.RenameNetworkInputTensor)==2);
                from = convertedLayer.RenameNetworkInputTensor(1);
                to = convertedLayer.RenameNetworkInputTensor(2);
                [tf, idx] = ismember(from, this.allNetInputNames);
                assert(tf);
                assert(isscalar(idx));
                this.allNetInputNames(idx) = to;
            end
        end

        function code = genCustomLayerCode(this)
            code = string.empty;
            if ~isempty(this.allCustomLayerNames) % || ~isempty(this.placeholderLayerCode)
                code = "## Helper layers:";
                if ~isempty(this.allCustomLayerNames)
                    code = [code
                        iIncludeCodeFiles(this.allCustomLayerNames + ".py")
                        ];
                end
            end
        end
    end
end

function loadModelCode = iGenDefineModelCode(inputNames, outputNames, layerCode)
defLine = "def create_model():";
assert(~isempty(inputNames));
assert(~isempty(outputNames));

modelLine = "model = keras.Model(inputs=[" + join(inputNames, ', ') + "], outputs=[" ...
    + join(outputNames, ', ') + "])";
returnLine = "return model";
loadModelCode = [defLine; iPrependTab(layerCode); ""; iPrependTab(modelLine); iPrependTab(returnLine); ""];
end

function strings = iPrependTab(strings)
% Add 4 spaces before each string
strings = "    " + strings;
end

function str = iFileCreatedByComment()
str = [
    "#    " + message('nnet_cnn_kerasimporter:keras_importer:exporterFileCreatedByCommentLine1').getString
    "#    " + message('nnet_cnn_kerasimporter:keras_importer:exporterFileCreatedByCommentLine2').getString
    "#    " + string(datetime)
    ""
    ];
end

function code = iGenUtilityFcnCode()
UtilityFcns = ["loadWeights.py"];
code = "## Utility functions:";
code = [code; iIncludeCodeFiles(UtilityFcns)];
end

function code = iIncludeCodeFiles(filenames)
rootDir = nnet.internal.cnn.tf_exporter.supportPackageRootDir();
code = "";
for i = 1:numel(filenames)
    fullPath = fullfile(rootDir, "toolbox", "nnet", "supportpackages", "keras_importer", "+nnet", "+internal", "+cnn", "+tf_exporter", filenames(i));
    fileID = fopen(fullPath);
    if fileID <= 0
        error(message('nnet_cnn_kerasimporter:keras_importer:exporterFileOpen', fullPath));
    end
    C = onCleanup(@()fclose(fileID));
    A = fread(fileID,'*char')';
    clear C
    code = [code; string(A); ""]; %#ok<AGROW>
end
end

function text = iWrapText(text)
text = char(text);
text = textwrap({text}, 76);
text = string(text);
end