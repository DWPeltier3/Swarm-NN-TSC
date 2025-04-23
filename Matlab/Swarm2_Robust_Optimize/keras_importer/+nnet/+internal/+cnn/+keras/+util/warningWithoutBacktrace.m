function warningWithoutBacktrace(msgID, varargin)
% warningWithoutBacktrace   Throw a warning suprressing the backtrace
% which contains internal functions. After throwing restore the initial
% state of the backtrace.

% Copyright 2018 The Mathworks, Inc.

backtrace = warning('query','backtrace');
warning('off','backtrace');
warning(message(msgID, varargin{:}));
warning(backtrace.state,'backtrace');
end