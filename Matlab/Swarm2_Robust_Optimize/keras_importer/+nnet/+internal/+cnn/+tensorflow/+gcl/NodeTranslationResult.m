classdef NodeTranslationResult < handle
%

%   Copyright 2020-2023 The MathWorks, Inc.

    properties
        Node                    % A TFNodeDef object which was translated
        Code                    % A string storing the generated code for a node
        ForwardRank  = false    % A boolean storing if the rank is unchanged by this operator
        NumOutputs              % A value storing the number of outputs that this operator creates
        OpFunctions             % A list of op functions that are needed for this node
        SubFunctions            % A string storing a subfunction's name if this Node calls one
        IsCommenting = false    % A boolean flag signaling if a comment is needed for this op
        Comment                 % A string storing a custom comment (Should not include the first '%') 
        Success                 % A boolean storing if the translation was successful. 
    end
    
    methods 
        function code = emitNode(result)
            code = "";
            % Optionally add a comment 
            if result.IsCommenting
                if isempty(result.Comment) || result.Comment == ""
                    code = string(newline) + "% Operation"; 
                else
                    code = string(newline) + "% " + result.Comment;
                end
            end
            
            % Append actual code
            code = code + newline + result.Code; 
        end
    end
end
