function [tf, Message] = canSupportSettingsConv(LSpec)
% canSupportSettingsConv  If the settings of the convolutional layer LSpec
% are supported

% Copyright 2018 The MathWorks, Inc.

if ~ismember(kerasField(LSpec, 'padding'), {'valid', 'same'})
    tf = false;
    Message = message('nnet_cnn_kerasimporter:keras_importer:UnsupportedPadding', LSpec.Name);
elseif ~isempty(kerasField(LSpec, 'activity_regularizer'))
    tf = false;
    Message = message('nnet_cnn_kerasimporter:keras_importer:NoActivityRegularization', LSpec.Name);
else
    tf = true;
    Message = '';
end
end
