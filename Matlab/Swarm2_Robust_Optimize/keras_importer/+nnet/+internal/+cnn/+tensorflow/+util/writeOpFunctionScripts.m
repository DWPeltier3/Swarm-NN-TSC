function writeOpFunctionScripts(opFunctionsUsed, customLayerPath, hasUnsupportedOp)
    % Copyright 2022-2023 The MathWorks, Inc.
    
    % Create the ops subpackage
    p = strsplit(customLayerPath, '+');
    packageName = p{end};
    opsPackage = [customLayerPath filesep '+ops'];
    spkgOpFileLocation = [fileparts(which('nnet.internal.cnn.tensorflow.importTensorflowLayers')) filesep 'op' filesep];
    if ~isfolder(opsPackage)
        mkdir(opsPackage);
    end
    
    % Copy util functions
    copyfile([spkgOpFileLocation 'sortByLabel.m'],opsPackage,'f');
    copyfile([spkgOpFileLocation 'sortToTFLabel.m'],opsPackage,'f');
    copyfile([spkgOpFileLocation 'permuteToReverseTFDimensionOrder.m'],opsPackage,'f');
    copyfile([spkgOpFileLocation 'iExtractData.m'],opsPackage,'f');
    copyfile([spkgOpFileLocation 'iAddDataFormatLabels.m'],opsPackage,'f');
    copyfile([spkgOpFileLocation 'iPermuteToForwardTF.m'],opsPackage,'f');
    opFunctionsUsed(end+1) = "addOutputLabel";
    opFunctionsUsed(end+1) = "permuteToTFDimensionOrder";
    opFunctionsUsed(end+1) = "iPermuteToReverseTF"; 
    
    % Write the opfunction scripts needed for the custom layers
    uniqueOpFunctionsUsed = unique(opFunctionsUsed);
    uniqueOpFunctionsUsed(cellfun('isempty',uniqueOpFunctionsUsed)) = [];
    for i = 1:numel(uniqueOpFunctionsUsed)
        % Read base op file
        opFileName = strcat(uniqueOpFunctionsUsed(i),".m"); 
        [fid,msg] = fopen(strcat(spkgOpFileLocation,opFileName));
        % If the support package is correctly installed we expect to
        % find the operator function template, throw an assertion failure if
        % that is not the case
        if fid == -1 && hasUnsupportedOp
            % unsupported operator in the model, this is replaced by a placeholder function
            % get placeholder function template 
            templateLocation = which('templatePlaceholderFunctionTensorFlow.txt');
            [fid,msg] = fopen(templateLocation, 'rt');
            
            % If the support package is correctly installed we expect to
            % find the placeholder function template, throw an error if
            % that is not the case
            % no unsupported operator in the model but function template is not found
            if fid == -1
                error(message('nnet_cnn_kerasimporter:keras_importer:PlaceholderFunctionTemplateNotFound', msg));
            end
            code = string(fread(fid, inf, 'uint8=>char')'); 
            fclose(fid);
            
            % Fill out the template
            code = strrep(code, "{{functionname}}", uniqueOpFunctionsUsed(i));
            code = strrep(code, "{{operatorname}}", extractBetween(opFileName,"tf",".m"));
            code = strrep(code, "{{functionfilename}}", opFileName);
           
        elseif fid == -1 && ~hasUnsupportedOp
            % no unsupported operator in the model but function template is not found
            error(message('nnet_cnn_kerasimporter:keras_importer:OpFunctionTemplateNotFound', opFileName, msg));
        else
            % supported operator in the model, use the function template to write op function script
            code = textscan(fid, '%s', 'Delimiter','\n', 'CollectOutput',true);
            code = code{1};
            fclose(fid);
        end        

        % add import line and write to ops subpackage
        code = strrep(code, "%{{import_statement}}", ['import ' packageName '.ops.*;']);
        
        [fid,msg] = fopen(strcat(opsPackage,filesep,opFileName), 'w');
        if fid == -1
			throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:UnableToCreateOpFunctionFile',opFileName,msg)));
        end
        code = nnet.internal.cnn.tensorflow.util.indentcode(strjoin(code,newline));
        fwrite(fid, code);
        fclose(fid);
    end
end