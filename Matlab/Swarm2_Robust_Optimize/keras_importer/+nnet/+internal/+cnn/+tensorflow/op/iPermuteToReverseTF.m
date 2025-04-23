function revTFInput = iPermuteToReverseTF(fwdTFInput, rank, isInternal)
%{{import_statement}}
%
% Permutes the data from forward to reverse TensorFlow format. The input
% data can either be a labeled or unformatted dlarray.
%
% isInternal flag determines if this function is being called from a
% top-level custom layer (False) or a nested custom layer (True).

%   Copyright 2023 The MathWorks, Inc.
    if nargin < 3
        isInternal = false;
    end

    if isdlarray(fwdTFInput) && rank >= 2
        inputLabels = fwdTFInput.dims;

        % Reverse if the input data is all U-labeled and the
        % isInternal is set to false.
        if ~isInternal && all(inputLabels == 'U')
            revTFInput = stripdims(permute(fwdTFInput, rank:-1:1));
        % Strip the labels for U-labeled dlarrays
        % if the isInternal flag is set to true.
        elseif isInternal && all(inputLabels == 'U')
            revTFInput = stripdims(fwdTFInput);
        % Leave as is for unformatted dlarrays
        % if the isInternal flag is set to true.
        elseif isInternal && isempty(inputLabels)
            revTFInput = fwdTFInput;
        % Permute DLT labels to unformatted dlarrays in reverse TF order
        else
            [permuteVec, ~] = sortToTFLabel(1:rank, inputLabels);
            revTFInput = permute(stripdims(fwdTFInput), flip(permuteVec));
        end
    else
        revTFInput = fwdTFInput;
    end
end