function y = tfAddN(varargin) 
%{{import_statement}}

%   Copyright 2020-2023 The MathWorks, Inc.

    y.value = varargin{1}.value; 
    y.rank = varargin{1}.rank; 

    for i = 2:nargin
        y = tfAdd(y, varargin{i}); 
    end
end
