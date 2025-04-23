function y = tfStatelessIf(cond)

%  cond : A Tensor. If the tensor is a scalar of
%         non-boolean type, the scalar is converted 
%         to a boolean according to the following rule: 
%         if the scalar is a numerical value, non-zero means True 
%         and zero means False; if the scalar is a string, non-empty means 
%         True and empty means False. If the tensor is not a scalar, being 
%         empty means False and being non-empty means True.
% 
%  Copyright 2022-2023 The MathWorks, Inc.
   
    condVal = cond.value;
    condRank = cond.rank;
   
    if condRank == 0
        % scalar condition
        if isnumeric(condVal) || islogical(condVal)
            if condVal ~= 0
                y = true;
            else
                y = false;
            end
        elseif isstring(condVal)
            if isempty(condVal)
                y = false;
            elseif (condVal == "")
                y = false;
            else
                y = true;
            end
        elseif ischar(condVal)
            if isempty(condVal)
                y = false;            
            else
                y = true;
            end
        end
        return
    else
        % non-scalar condition
        if isempty(condVal)
            y = false;            
        else
            y = true;
        end
    end
end
