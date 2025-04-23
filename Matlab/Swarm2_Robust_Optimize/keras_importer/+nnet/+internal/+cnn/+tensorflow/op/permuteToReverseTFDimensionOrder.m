function [varargout] = permuteToReverseTFDimensionOrder(varargin)
% Copyright 2023 The MathWorks, Inc.
% PERMUTETOREVERSETFDIMENSIONORDER This function permutes placeholder function outputs from 
% forward TF dimension order to reverse TF Dimension order 
    varargout = cell(1, nargin);
    for i=1:nargin
        x = varargin{i};
        if isstruct(x)
            % input is a struct with 'value' and 'rank' fields.    
            if x.rank > 1
                x.value = permute(dlarray(x.value), x.rank:-1:1);     
            elseif x.rank <= 1
                x.value = dlarray(x.value);
            end
            varargout{i} = x;
        else            
            % return the value unchanged;
            varargout{i} = x;
        end
    end    
end