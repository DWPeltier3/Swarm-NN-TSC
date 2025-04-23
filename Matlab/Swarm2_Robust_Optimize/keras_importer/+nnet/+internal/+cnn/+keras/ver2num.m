function n = ver2num(KerasVersion)
% KerasVersion is a string such as '2.3.3a...', where 'a' is a non-digit
% and ... is of unknown length. Output would be 20303.

% Extract numbers separated by non-numbers
C = str2double(regexp(KerasVersion, '\d*', 'match'));
% Extend to 3 numbers
switch numel(C)
    case 1
        C(2:3) = [0 0];
    case 2
        C(3) = 0;
end
% Limit to 3 numbers
C = C(1:3);
% Convert to single number
n = [10000 100 1]*C';
end