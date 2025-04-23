classdef TFConstants < handle
    % Constants object that stores imported TF constants.

%   Copyright 2020-2021 The MathWorks, Inc.

    properties 
        Constants
        ConstantsRank
        NumDuplicates
    end
    
    methods
        function obj = TFConstants(nameMap)
            obj.NumDuplicates = struct;
            obj.Constants = struct;
            obj.ConstantsRank = struct;
            if nargin > 0
                for curName = nameMap.values
                    curVar = curName{1}; 
                    % Just mark this as a duplicate but we dont want this
                    % to be generated. 
                    obj.NumDuplicates.(curVar) = 0; 
                    %obj.Constants.(curVar) = 0; 
                    %obj.ConstantsRank.(curVar) = 0; 
                end
            end

        end
        
        function constName = addConstant(this, name, value, rank)
            if numel(name) > namelengthmax
                name = name(:, 1:namelengthmax);
            end

            if ~isfield(this.NumDuplicates, name)
                this.NumDuplicates.(name) = 0; 
                this.Constants.(name) = value; 
                this.ConstantsRank.(name) = rank;
                constName = name; 
            else
                % This constant name was already used before.
                this.NumDuplicates.(name) = this.NumDuplicates.(name) + 1;
                postfix = num2str(this.NumDuplicates.(name));
                
                % Support up to 99 duplicates.
                maxLen = 60;
                
                % Truncate the right part of the original identifier to
                % make enough for the postfix. Include the '_' to guarantee
                % uniqueness. 
                cutoffIdx = max(maxLen, namelengthmax - (numel(postfix) + 1));
                truncated = name(1:min(cutoffIdx, numel(name)));
                constName = [truncated '_' postfix];
                
                this.Constants.(constName) = value;
                this.ConstantsRank.(constName) = rank;
            end
        end

        function constName = updateConstant(this, name, value, rank)
            this.Constants.(name) = value; 
            this.ConstantsRank.(name) = rank;
            constName = name; 
        end
        
        function variableNames = getConstNames(this)
            variableNames = fields(this.Constants); 
        end
        
        function [values, rank] = lookupConst(this, constName) 
            values = this.Constants.(constName); 
            rank = this.ConstantsRank.(constName); 
        end 
    end
    
    methods (Hidden)
        function reset(this)
            % Clear all stored constants. 
            this.NumDuplicates = struct;
            this.Constants = struct;
            this.ConstantsRank = struct;
        end
    end
end

