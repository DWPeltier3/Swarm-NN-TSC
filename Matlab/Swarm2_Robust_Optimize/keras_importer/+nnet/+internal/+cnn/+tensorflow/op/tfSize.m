function sz = tfSize(in)

%   Copyright 2022-2023 The MathWorks, Inc.
    inVal = in.value;
    
    % Evaluate size of the input tensor:
    sz = dlarray(numel(inVal));
    
    % assign output rank, which is always 0 
    sz = struct('value', sz, 'rank', 0);
end
