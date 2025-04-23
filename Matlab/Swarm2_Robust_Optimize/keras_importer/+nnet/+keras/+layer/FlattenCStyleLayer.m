classdef FlattenCStyleLayer < nnet.layer.Layer & nnet.layer.Formattable
% FlattenCStyleLayer   FlattenCStyle layer
%
% layer = FlattenCStyleLayer(Name) creates a layer with name Name that
% flattens the input image into a vector, assuming C-style (or row-major)
% storage ordering of the input layer.
% 
% For 2D inputs, the Keras Flatten layer has input shape (with
% channels_last) [N H W C] and output shape [N HWC], where HWC have been
% flattened in that order and C-style. In MATLAB the input tensor has shape
% [H W C N] and we want the output to be [1 1 HWC N] with HWC flattened in
% that order and FORTRAN-style. So we want to create [H W C N] in MATLAB
% where, for each N, the HWC are in row-major order, and then reshape. [H W
% C] in col-major comes in, which is the same order as [C W H] row-major.
% So we just need to permute the first 3 dimensions to turn HWC into CWH.
% That's [3 2 1 4]. In the backward pass we reverse the process, first
% reshaping dLdZ to [C W H N], then undoing the permutation, which happens
% to be [3 2 1 4] again.
%
% For 3D inputs, the Keras Flatten layer has input shape (with
% channels_last) [N H W D C] and output shape [N HWDC], where HWDC have
% been flattened in that order and C-style. In MATLAB the input tensor has
% shape [H W D C N] and we want the output to be [1 1 HWDC N] with HWDC
% flattened in that order and FORTRAN-style. So we want to create [H W D C
% N] in MATLAB where, for each N, the HWDC are in row-major order, and then
% reshape. [H W D C] in col-major comes in, which is the same order as [C D
% W H] row-major. So we just need to permute the first 4 dimensions to turn
% HWDC into CDWH. That's [4 3 2 1 5]. In the backward pass we reverse the
% process, first reshaping dLdZ to [C D W H N], then undoing the
% permutation, which happens to be [4 3 2 1 5] again.
% 
% For 1D inputs with Time Dimension, the Keras flatten layer has input
% shape (with channels_last) [N T C] and output shape [N TC], where TC have
% been flattened in that order and C-style. In MATLAB the input tensor has
% shape [C N T] and we want the output to be [TC N] with TC flattened in
% that order and FORTRAN-style. So we want to create [C N T] in MATLAB
% where, for each N, the TC are in row-major order, and then reshape.[C N T]
% in col-major comes in, which is the same order as [T N C] row major. So
% we just need to permute the second and third dimension to turn [C N T] to
% [C T N]. That's [1 3 2].
    
    %   Copyright 2017-2023 The MathWorks, Inc.
    methods
        function this = FlattenCStyleLayer(name)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenDescription'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenType'));
        end
        
        function Z = predict( ~, X )
            fmt = dims(X);
            if ismember(fmt, ["SSSC", "SSSCB"])
                % X is size [H W C D N].
                % Z is size [1 1 1 HWCD N].
                [sz1, sz2, sz3, sz4, sz5] = size(X);
                Z = reshape(permute(stripdims(X),[4 3 2 1 5]), [1 1 1 sz1*sz2*sz3*sz4 sz5]);
                Z = dlarray(Z, fmt);
            elseif ismember(fmt, ["SC", "SCB"])
                [sz1, sz2, sz3] = size(X); 
                Z = reshape(permute(stripdims(X), [2 1 3]), [1 sz1*sz2 sz3] ); 
                Z = dlarray(Z, fmt); 
            elseif ismember(fmt,"CBT")
                %Flatten with Sequence Input Layer
                %Common use case for 1D Layers
                %X: [C B T] ; Z:[CT B]
                [sz1, sz2, sz3] = size(X);
                Z = reshape(permute(stripdims(X),[1 3 2]), [sz1*sz3 sz2]);
                Z = dlarray(Z, "CB");
            elseif ismember(fmt, ["SSC", "SSCB", "SSCT"])
                % X is size [H W C N].
                % Z is size [1 1 HWC N].
                [sz1, sz2, sz3, sz4] = size(X);
                Z = reshape(permute(stripdims(X),[3 2 1 4]), [1 1 sz1*sz2*sz3 sz4]);
                Z = dlarray(Z, fmt);
            elseif ismember(fmt,"CB")
                Z = X;
            else
                % Error if X has unexpected data format
                throw(MException('nnet_cnn_kerasimporter:keras_importer:FlattenCStyleUnexpectedInput', ...
                    message('nnet_cnn_kerasimporter:keras_importer:FlattenCStyleUnexpectedInput', 'FlattenCStyleLayer')));
            end
        end
    end
end

