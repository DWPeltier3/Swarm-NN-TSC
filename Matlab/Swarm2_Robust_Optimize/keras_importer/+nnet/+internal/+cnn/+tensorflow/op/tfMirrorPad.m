function y = tfMirrorPad(input, paddings, mode)

%   Copyright 2020-2023 The MathWorks, Inc.

    inputRank = input.rank;
    inputVal = input.value;

    if ~isdlarray(inputVal)
        % If a numeric array permute the input tensor to match Forward TF Format.   
        inputVal = permute(inputVal, inputRank:-1:1);      
    elseif inputRank > 1
        % Assume the input dlarray to be in reverse TF order and permute to
        % forward TF
        inputVal = permute(inputVal.stripdims, inputRank:-1:1);
    else
        % dlarray with rank <= 1, is already in forward Tf format
        inputVal = inputVal.stripdims;
    end
        
    % paddings is a Nx2 matrix. Where N is the number of dimensions of x. 
    % For a dimension D paddings(D, 1) is the number of values to pad
    % before the contents of x in that dimension. and paddings(D, 2) is the
    % number of values to add after the contents of x in that dimension. 
    paddings = paddings.value'; 
    sizeX = size(inputVal); 
    if inputRank > numel(sizeX)
        % add back potentially dropped dims 
        diff = inputRank - numel(sizeX); 
        sizeX(end+1:end+diff) = 1; 
    end 
    sizeY = zeros(1, inputRank); 
    for i = 1:size(paddings, 1)
        sizeY(i) = sizeX(i) + paddings(i, 1) + paddings(i, 2); 
    end 
    
    % Construct output array with padded size. 
    yVal = dlarray(zeros(sizeY, 'like', inputVal));
    
    % Construct subsref indices for inserting (and cropping) the original
    ySubs = cell(1, size(paddings, 1)); 
    xSubs = cell(1, size(paddings, 1)); 
    
    for i=1:numel(sizeX)
        ySubs{i} = max(1,1+paddings(i,1)) : min(sizeY(i), sizeY(i)-paddings(i, 2));
        xSubs{i} = max(1,1-paddings(i,1)) : min(sizeX(i), sizeX(i)+paddings(i, 2));
    end
    sY      = struct('type', '()');
    sY.subs = ySubs;
    sX      = struct('type', '()');
    sX.subs = xSubs;
    
    % Insert/crop the original into the result
    yVal = subsasgn(yVal, sY, subsref(inputVal, sX));
    
    % Handle 'reflect' and 'symmetric' modes
    mode = upper(mode); 
    if ismember(mode, ["REFLECT" "SYMMETRIC"])
        for dim = 1:inputRank
            if any(paddings(dim, :) > 0)
                % Setup a call to subsasgn
                prepad  = paddings(dim, 1);
                postpad = paddings(dim, 2);
                if prepad > 0
                    [sY, sX] = prepadIndices(sizeX, prepad, dim, mode);
                    yVal = subsasgn(yVal, sY, subsref(yVal, sX));
                end
                if postpad > 0
                    [sY, sX] = postpadIndices(sizeX, sizeY, prepad, postpad, dim, mode);
                    yVal = subsasgn(yVal, sY, subsref(yVal, sX));
                end
            end
        end
    else
        % Unrecognized padding mode. 
        error('Unsupported padding mode. Only REFLECT and SYMETRIC modes are supported for MirrorPad.'); 
    end
    
    % re-label and convert back to DLT / Reverse TF ordering. 
    if inputRank > 1
        % Convert to reverse-TF format
        yVal = permute(yVal, inputRank:-1:1);
    end
    
    yVal = dlarray(yVal);
    y = struct('value', yVal, 'rank', inputRank);
    
    % Subfunctions in tfMirrorPad:
    function [Sy, Sx] = prepadIndices(sizeX, prepad, dim, mode)
        Sy   	= struct('type', '()');
        Sy.subs	= repmat({':'}, [1 numel(sizeX)]);
        Sx   	= Sy;
        % Write into the first 'prepad' elements of Y.dim.
        Sy.subs{dim} = 1:prepad;
        switch mode
            case 'REFLECT'
                % Create indices 2:prepad+1 of X.dim, in the reverse order, with
                % wraparound. Then add prepad to convert them to Y indices.
                Sx.subs{dim} = wrapIndices(prepad+1 : -1 : 2, sizeX(dim)) + prepad;
            case 'SYMMETRIC'
                % Create indices 1:prepad of X.dim, in the reverse order, with
                % wraparound. Then add prepad to convert them to Y indices.
                Sx.subs{dim} = wrapIndices(prepad : -1 : 1, sizeX(dim)) + prepad;
        end
    end

    function [Sy, Sx] = postpadIndices(sizeX, sizeY, prepad, postpad, dim, mode)
        Sy   	= struct('type', '()');
        Sy.subs	= repmat({':'}, [1 numel(sizeX)]);
        Sx   	= Sy;
        % Write into the last 'postpad' elements of Y.dim.
        Sy.subs{dim} = sizeY(dim)-postpad+1 : sizeY(dim);
        switch mode
            case 'REFLECT'
                % Create indices in the reverse order, with wraparound. Then add
                % prepad to convert them to Y indices.
                Sx.subs{dim} = wrapIndices(sizeX(dim)-1 : -1 : sizeX(dim)-postpad, sizeX(dim)) + prepad;
            case 'SYMMETRIC'
                % Create indices in the reverse order, with wraparound. Then add
                % prepad to convert them to Y indices. Include the end index. 
                Sx.subs{dim} = wrapIndices(sizeX(dim) : -1 : sizeX(dim)-postpad + 1, sizeX(dim)) + prepad;
        end
    end

    function j = wrapIndices(i, maxIdx)
        % i can be positive, negative or zero. Legal output indices are in the
        % range 1:maxIdx.
        j = mod(i-1, maxIdx) + 1;
    end
end 

