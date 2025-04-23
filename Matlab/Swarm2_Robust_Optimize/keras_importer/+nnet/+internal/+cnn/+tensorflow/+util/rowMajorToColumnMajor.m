function X = rowMajorToColumnMajor(X, numdims)
%

%   Copyright 2020-2021 The MathWorks, Inc.

    DimVec = ones(1, numdims); 
    DimVec(1:ndims(X)) = size(X); 
    if numdims > 1
        DimVec = DimVec(:)';            % Make sure dim is a row vector.
        X = reshape(X, fliplr(DimVec));         % Declare X to have the reversed given shape (because it's row-major).
    end
end
