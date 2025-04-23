classdef inputVerificationLayer < nnet.layer.Layer & nnet.layer.Formattable 
    % inputVerificationLayer   Verify input size and input data format
    %
    % v = inputVerificationLayer(NAME,SIZE,FORMAT) creates an input verification
    % layer and sets the Name, Size, and Format properties. This v layer 
    % verifies the input to the NAME layer has the size SIZE and data format 
    % FORMAT. An inputVerificationLayer layer returns a warning if the input 
    % size does not match the Size property and returns an error if the input 
    % data format does not match the Format property.
    
    %   Copyright 2022 The MathWorks, Inc.

    properties
        Size
        Format
    end

    methods
        function obj = inputVerificationLayer(Name, Size, Format)
            arguments
                Name {mustBeTextScalar}
                Size {mustBeNumeric, mustBeNonempty, mustBeReal , ...
                mustBeFinite, mustBeInteger, mustBePositive}
                Format {mustBeTextScalar}
            end
            obj.Name = Name;
            obj.Size = Size;
            obj.Format = Format;  
            obj.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:InputVerificationType'));
            obj.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:InputVerificationDescription',Name, num2str(Size), Format));
        end

        function Z = predict(obj, X)
            [expectedFmt, idx] = nnet.internal.cnn.tensorflow.util.sortToDLTLabel(obj.Format);
            expectedSize = obj.Size(idx);
            if ~(iVerifyFormat(X, expectedFmt)) && ~all(expectedFmt == 'U')
                % Throw an Exception if the input format and the expected
                % formats don't match and the expected format is fully
                % known (i.e. Not all 'U's).
                throw(MException('nnet_cnn_kerasimporter:keras_importer:InputVerificationMismatch', iDisplayMismatchInfo(obj.Name, X, expectedSize, expectedFmt)));
            elseif ~(iVerifyFormat(X, expectedFmt)) && all(expectedFmt == 'U')
                % Only throw a warning when the expectedFmt is all 'U's
                % (i.e. Unknown format case) and there is a format mismatch
                nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:InputVerificationMismatch', iDisplayMismatchInfo(obj.Name, X, expectedSize, expectedFmt));
            end

            if ~(iVerifySize(X, expectedSize, expectedFmt)) 
                % Throw a warning if the input size and the expected size
                % don't match
                nnet.internal.cnn.keras.util.warningWithoutBacktrace('nnet_cnn_kerasimporter:keras_importer:InputVerificationMismatch', iDisplayMismatchInfo(obj.Name, X, expectedSize, expectedFmt));
            end
            Z = X;
        end
    end
end

function tf = iVerifyFormat(X, expectedFmt)
    fmt = X.dims;
    diffRank = numel(expectedFmt) - numel(fmt);
    if ~diffRank
        tf = all(fmt == expectedFmt,'all');
    elseif diffRank && contains(expectedFmt, fmt)
        updatedFmt = [fmt 'U'*(diffRank)];
        tf = all(updatedFmt == expectedFmt,'all');    
    else
        tf = false;
    end
end

function tf = iVerifySize(X, expectedSize, expectedFmt)
    sizeMatchIdx = iFindSizeIdxToMatch(expectedFmt);
    tf = all(X.size(sizeMatchIdx) == expectedSize(sizeMatchIdx),'all');
end

function idx = iFindSizeIdxToMatch(fmt)
    dimB = find(fmt == 'B');
    dimU = find(fmt == 'U');
    dimT = find(fmt == 'T');
    idx = setdiff(1:numel(fmt), [dimT dimB dimU]);
end

function msg = iDisplayMismatchInfo(name, X, expectedSize, expectedFmt)
    prologue = getString(message('nnet_cnn_kerasimporter:keras_importer:InputVerificationMismatchPrologue', name));
    expected = getString(message('nnet_cnn_kerasimporter:keras_importer:InputVerificationExpected',name, num2str(expectedSize), expectedFmt, num2str(X.size), X.dims));
    msg = [prologue newline expected];        
end