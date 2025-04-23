classdef FormatConverter

    %   Copyright 2022 The MathWorks, Inc.

    % A class to convert between corresponding DLT and Tensorflow data
    % formats
    properties
        DLTFormats = [
            % Feature
            "BC"            % BC appears only as the output of a DAGNetwork output layer.
            "CB"
            "CU"            % With U=1. This occurs when you pass CT to globalAveragePooling1dLayer. It can also be passed directly into a dlnetwork.
            % Spatial
            "SC"            % dlnetwork only
            "SSC"           % dlnetwork only
            "SSSC"          % dlnetwork only
            "SCB"
            "SSCB"
            "SSSCB"
            "SSSSCB"        % For maxPooling2DLayer's "size" output.
            % Temporal
            "CT"            % dlnetwork only
            "CBT"
            % Spatial & temporal
            "SCT"           % dlnetwork only
            "SSCT"          % dlnetwork only
            "SSSCT"         % dlnetwork only
            "SCBT"
            "SSCBT"
            "SSSCBT"
            % Unknown
            "UU"            % dlnetwork only
            "UUU"           % dlnetwork only
            "UUUU"          % dlnetwork only
            ];
        TFFormats = [
            % Feature
            "BC"
            "BC"
            "BC"
            % Spatial
            "BSC"
            "BSSC"
            "BSSSC"
            "BSC"
            "BSSC"
            "BSSSC"
            "BSSSSC"        % For maxPooling2DLayer's "size" output.
            % Temporal
            "BTC"
            "BTC"
            % Spatial & temporal
            "BTSC"
            "BTSSC"
            "BTSSSC"
            "BTSC"
            "BTSSC"
            "BTSSSC"
            % Unknown
            "BC"
            "BSC"
            "BSSC"
            ];
    end

    methods (Access=protected)
        function perm = permToTF(this, mlFormat)
            % Given a DLT format string, return the permutation vector
            % required to permute the matlab dimension ordering into the
            % corresponding TF ordering. Does not include the
            % transformation to row-major storage ordering.
            switch mlFormat
                % Feature
                case "BC"            % BC appears only as the output of a DAGNetwork output layer.
                    perm = [1 2];
                case {"CB", "CU"}
                    perm = [2 1];
                    % Spatial
                case "SC"
                    perm = [3 1 2];
                case "SSC"
                    perm = [4 1 2 3];
                case "SSSC"
                    perm = [5 1 2 3 4];
                case "SCB"
                    perm = [3 1 2];
                case "SSCB"
                    perm = [4 1 2 3];
                case "SSSCB"
                    perm = [5 1 2 3 4];
                case "SSSSCB"
                    perm = [6 1 2 3 4 5];   % For maxPooling2DLayer's "size" output.
                    % Temporal
                case "CT"
                    perm = [3 2 1];
                case "CBT"
                    perm = [2 3 1];
                    % Spatial + temporal
                case "SCT"
                    perm = [4 3 1 2];
                case "SSCT"
                    perm = [5 4 1 2 3];
                case "SSSCT"
                    perm = [6 5 1 2 3 4];
                case "SCBT"
                    perm = [3 4 1 2];
                case "SSCBT"
                    perm = [4 5 1 2 3];
                case "SSSCBT"
                    perm = [5 6 1 2 3 4];
                    % Unknown
                case "UUUU"
                    perm = [4 3 2 1];   % For selfAttentionLayer's scores output
                otherwise
                    assert(false);
            end
        end
    end

    methods (Static)
        function tfArray = permuteArrayToTF(mlArray, mlFormat)
            % Permutes a matlab array into the corresponding TF dimension
            % ordering, but does not transform to row-major storage
            % ordering
            assert(strlength(mlFormat)>=ndims(mlArray))
            this = nnet.internal.cnn.tf_exporter.FormatConverter;
            perm = permToTF(this, mlFormat);
            tfArray = permute(mlArray, perm);
        end

        %         function [tfArray, tfShape] = permuteArrayToTF(mlArray, mlFormat)
        %             % Given a matlab array with known format string, permute the
        %             % data to TF data ordering and return the TF shape. For
        %             % example, for input size [5 7 9] and "SCT", the output is a
        %             % BTSC array in TF, so tfshape is [1 9 5 7]. The data is
        %             % permuted so as to produce reverse-TF ordering in matlab:
        %             % permute(mlArray, [2 1 3 4]) to transform SCT --> CSTB.
        %             this = nnet.internal.cnn.tf_exporter.FormatConverter;
        %             perm = permToReverseTF(this, mlFormat);
        %             tfArray = permute(mlArray, perm);
        %             tfShape = fliplr(size(tfArray, 1:numel(perm)));
        %         end

        function tfFormats = mlFormatToTfFormat(mlFormats)
            % Given a string array of DLT formats, return the corresponding
            % TF format strings.
            this = nnet.internal.cnn.tf_exporter.FormatConverter;
            [tf, pos] = ismember(mlFormats, this.DLTFormats);
            assert(all(tf))
            tfFormats = this.TFFormats(pos);
        end


        function tfDim = mlDimToTFDim(mlFormat, mlDim)
            % Given a dimension number of a DLT format (origin-1), return
            % the index (origin-0) of the corresponding dimension in the TF
            % format.
            assert(mlDim>0 && mlDim <= strlength(mlFormat))
            switch mlFormat
                % Feature
                case "BC"            % BC appears only as the output of a DAGNetwork output layer.
                    tfDim = mlDim-1;
                case {"CB", "CU"}
                    tfDim = 2-mlDim;
                    % Spatial
                case {"SC", "SSC", "SSSC"}
                    tfDim = mlDim;
                case "SCB"
                    tfDim = mod(mlDim, 3);
                case "SSCB"
                    tfDim = mod(mlDim, 4);
                case "SSSCB"
                    tfDim = mod(mlDim, 5);
                case "SSSSCB"
                    tfDim = mod(mlDim, 6);  % For maxPooling2DLayer's "size" output.
                    % Temporal
                case "CT"
                    tfDim = 3-mlDim;
                case "CBT"
                    tfDim = mod((mlDim+1), 3);
                    % Spatial + temporal
                case "SCT"
                    tfDim = 1 + mod(mlDim, 3);
                case "SSCT"
                    tfDim = 1 + mod(mlDim, 4);
                case "SSSCT"
                    tfDim = 1 + mod(mlDim, 5);
                case "SCBT"
                    tfDim = mod(mlDim+1, 4);
                case "SSCBT"
                    tfDim = mod(mlDim+1, 5);
                case "SSSCBT"
                    tfDim = mod(mlDim+1, 6);
                    % Unknown
                otherwise
                    assert(false);
            end
        end
    end
end