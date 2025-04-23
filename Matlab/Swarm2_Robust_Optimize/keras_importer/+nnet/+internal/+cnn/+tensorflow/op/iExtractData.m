function x = iExtractData(x)
    if ~isa(x,'dlarray')
        x = dlarray(x);
    end
end