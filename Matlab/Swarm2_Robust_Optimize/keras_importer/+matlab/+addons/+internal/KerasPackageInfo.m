classdef KerasPackageInfo < matlab.addons.internal.SupportPackageInfoBase
    % KTensorFlow-keras support package support for MATLAB Compiler.
    
    %   Copyright 2020 The MathWorks, Inc.
    
    methods
        function obj = KerasPackageInfo()
            obj.baseProduct = 'Deep Learning Toolbox';
            obj.displayName = 'Deploy Imported Models for Deep Learning Toolbox Converter for TensorFlow Models';
            obj.name        = 'Deep Learning Toolbox Converter for TensorFlow Models';
            
            sproot = matlabshared.supportpkg.getSupportPackageRoot();
            
            % Define all the data that should be deployed from the support
            % package. This includes the actual language data, which will
            % be archived in the CTF.
            obj.mandatoryIncludeList = {...
                fullfile(sproot, 'toolbox','nnet','supportpackages','keras_importer','+nnet','+keras','+layer')};
         end
    end
end