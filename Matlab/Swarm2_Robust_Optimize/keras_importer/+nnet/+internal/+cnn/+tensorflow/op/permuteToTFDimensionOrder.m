function [varargout] = permuteToTFDimensionOrder(varargin)
% Copyright 2023 The MathWorks, Inc.
% PERMUTETOTFDIMENSIONORDER This function permutes placeholder function inputs to forward TF Dimension order 
    varargout = cell(1, nargin);
    for i=1:nargin
        x = varargin{i};
        if isstruct(x)
            % input is a struct with 'value' and 'rank' fields.    
            if x.rank > 1
                x.value = permute(x.value, x.rank:-1:1);
            end
            varargout{i} = x;
        else            
            % return the value unchanged;
            varargout{i} = x;
        end
    end  
end