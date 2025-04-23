classdef ImportIssue
    %IMPORTISSUE An object describing a problem that occurred during 
    %import of a TensorFlow-Keras model

    %   Copyright 2023 The MathWorks, Inc        
    
    properties
        Operator
        LayerClass
        MessageID
        MessageArgs
        Placeholder
    end
    
    methods
        function obj = ImportIssue(args)
            arguments
                args.Operator = string.empty;
                args.MessageID = string.empty;
                args.MessageArgs = {};
                args.Placeholder = false;
                args.LayerClass = "";
            end  
            obj.Operator = args.Operator;
            obj.LayerClass = args.LayerClass;
            obj.MessageID = args.MessageID;
            obj.MessageArgs = args.MessageArgs;
            obj.Placeholder = args.Placeholder;            
        end

        function msg = getMessage(obj)
            msg = message.empty;
            if ~isempty(obj.MessageID)
                if ~isempty(obj.MessageArgs)
                    msg = message(obj.MessageID, obj.MessageArgs{:});
                else
                    msg = message(obj.MessageID);
                end
            end
        end

        function msgString = getMessageString(obj)
            msg = getMessage(obj);
            msgString = '';            
            if ~isempty(msg)
                msgString = getString(msg);
            end
        end

        function warningWithoutBacktrace(obj)
            % warningWithoutBacktrace   Throw a warning suprressing the backtrace
            % which contains internal functions. After throwing restore the initial
            % state of the backtrace.
            if ~isempty(obj.MessageID)
                backtrace = warning('query','backtrace');
                warning('off','backtrace');
                if ~isempty(obj.MessageArgs)
                    warning(message(obj.MessageID, obj.MessageArgs{:}));
                else
                    warning(message(obj.MessageID));
                end
                warning(backtrace.state,'backtrace');
            end
        end
    end  
end