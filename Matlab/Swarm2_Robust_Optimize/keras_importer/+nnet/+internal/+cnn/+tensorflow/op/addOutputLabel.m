function [output] = addOutputLabel(output, outputNum, layerObj)
%{{import_statement}}

% This function adds data format labels to dlarrays in 'output'

%   Copyright 2022-2023 The MathWorks, Inc.
    
    dlOutRank = output.rank;
    if ~isdlarray(output.value)   
        if dlOutRank > 1
            output.value = dlarray(output.value, repmat('U', [1 dlOutRank]));
        else
            output.value = dlarray(output.value, 'UU');
        end         
    end

    dlOut = output.value;
    dlOutLabels = dlOut.dims;
    
    if numel(layerObj.OutputLabels) >= outputNum
        fwdTFLabel = layerObj.OutputLabels{outputNum};
        if strcmp(fwdTFLabel,'t')
            % input going to a Time distributed layer
            % encapsulating a custom layer, 
            % we cannot determine a label in this case
            fwdTFLabel = '';
        end
    else
        % unused output, no need to label it
        fwdTFLabel = '';
    end
    
    if ~isempty(fwdTFLabel) && (isempty(dlOutLabels) || all(dlOutLabels == 'U'))
       
        if strcmp(fwdTFLabel,'tdo')
            % This is a GCL encapsulated by sequence folding and unfolding layers
            % in this case we give precedence to its own input labels over 
            % its successors input labels and try to re-apply own input labels first in this case
            inputLabels = layerObj.InputLabels{1};
            if ~isempty(inputLabels) && ~all(inputLabels == 'U')
                [~, fwdTFLabel] = sortToTFLabel(1:dlOutRank, inputLabels);
                if (numel(fwdTFLabel) - dlOutRank) == numel(extract(fwdTFLabel, 'S'))
                    % Flatten inside TimeDistributedWrapper collapses the
                    % spatial dimensions
                    fwdTFLabel = erase(fwdTFLabel, 'S');
                end 
            else
                output.value = dlarray(dlOut, repmat('U', 1, max(dlOutRank, 2)));
                return;
            end
        end
        
        if strcmp(fwdTFLabel,'fcs')
        % Output labels not found due to a blocking FlattenCStyleLayer
        % BSS* case
            trailingSLabels = char('S' + zeros(1, dlOutRank - 1));
            fwdTFLabel = ['B' trailingSLabels];
        elseif strcmp(fwdTFLabel,'fcst')
        % Output labels not found due to a blocking FlattenCStyleLayer
        % BTSS* case with 
            trailingSLabels = char('S' + zeros(1, dlOutRank - 2));
            fwdTFLabel = ['BT' trailingSLabels];
        elseif strcmp(fwdTFLabel,'sm')
        % Output labels not found due to a softmax layer
        % BU*C case
            midULabels = char('S' + zeros(1, dlOutRank - 2));
            fwdTFLabel = ['B' midULabels 'C']; 
        elseif strcmp(fwdTFLabel,'smt')
        % Output labels not found due to a softmax layer
        % BTU*C case
            midULabels = char('S' + zeros(1, dlOutRank - 3));
            fwdTFLabel = ['BT' midULabels 'C']; 
        elseif strcmp(fwdTFLabel,'classregout')
        % Output labels not found due to a regression/classification output layer
        % BTC/ BSSC/ BTSSC case
            switch dlOutRank
                case 2 
                    midLabels = '';
                case 3
                    midLabels = 'T';
                case 4
                    midLabels = 'SS';
                case 5
                    midLabels = 'TSS';
                otherwise
                    midLabels = char('S' + zeros(1, dlOutRank - 2));
            end
            fwdTFLabel = ['B' midLabels 'C'];

        elseif strcmp(fwdTFLabel,'pixelout') %|| strcmp(fwdTFLabel,'bce')
        % Output labels not found due to a pixelclassification 
        % or BinaryCrossEntropyRegressionLayer output layer BC / BSSC / BSSSC case
            midSLabels = char('S' + zeros(1, dlOutRank - 2));
            fwdTFLabel = ['B' midSLabels 'C'];
        end

        if dlOutRank > 1 
            if dlOutRank > numel(fwdTFLabel)
            % Extra U labelled batch dims
               augULabels = char('U' + zeros(1, dlOutRank - numel(fwdTFLabel)));
               fwdTFLabel = ['B' augULabels fwdTFLabel(2:end)];
            elseif dlOutRank < numel(fwdTFLabel)
            % More labels expected than tracked rank / data
            % Rank is not tracked, most likely because of a TFOpLambda layer
            % Increase output rank to match number of expected labels
                dlOutRank = numel(fwdTFLabel);
            end
            % Permute dlOut to forward TF format and apply TF labels
            dlOut = permute(dlOut, dlOutRank:-1:1);
        end

        %  dlOut should be in forward TF format at this point
        output.value = dlarray(dlOut, fwdTFLabel);
    elseif isempty(fwdTFLabel) && ismember(layerObj.Type, {'Dense'}) && (isempty(dlOutLabels) || all(dlOutLabels == 'U'))
        % This is a GCL for Dense layer for which we do not have output labels
        % re-apply input labels in this case
        inputLabels = layerObj.InputLabels{1};
        if ~isempty(inputLabels) && ~all(inputLabels == 'U')
            [~, fwdTFLabel] = sortToTFLabel(1:dlOutRank, inputLabels);
            if dlOutRank > 1
                % Permute dlOut to forward TF format and apply TF labels
                dlOut = permute(dlOut, dlOutRank:-1:1);
            end
            %  dlOut should be in forward TF format at this point
            output.value = dlarray(dlOut, fwdTFLabel);
        end
    elseif ~strcmp(layerObj.Type, 'SubclassedModel') && isempty(fwdTFLabel) && (isempty(dlOutLabels) || all(dlOutLabels == 'U')) && dlOutRank == 2
        % Assign CB labels as these are required for a keras layer
        output.value = dlarray(dlOut, 'CB');
    elseif isempty(fwdTFLabel) && isempty(dlOutLabels)
        % Unlabeled dlarray in rev TF format
        output.value = dlarray(dlOut, repmat('U', 1, max(dlOutRank, 2)));
    end 
end
