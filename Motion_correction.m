%The images are motion corrected
if ~exist (fullfile([folder_name_wr1,'_rig.mat']))
    disp('Performing Registration');
    options_rg = NoRMCorreSetParms('d1',512,'d2',512,'init_batch',200,...
        'bin_width',200,'max_shift',24,'tiff_filename',fullfile([folder_name_wr1,'_rig.mat']));
    [M_rg,shifts_rg,template_rg] = normcorre_batch(data,options_rg);
    data_reg = matfile(fullfile([folder_name_wr1,'_rig.mat']));
    data_reg.M_rg=M_rg;
    
else
    disp ('Already registered...Loading registered Data')
    load (fullfile([folder_name_wr1,'_rig.mat']))
end
clear data;
