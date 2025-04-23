function varargout = tfIdentityN(varargin)

%   Copyright 2020 The MathWorks, Inc.

    varargout = cell(nargin, 1); 
    for i = 1:nargin
        varargout{i} = varargin{i}; 
    end 
end 
