%Raw data files from same field are taken as input.
%All the image files are combined to a single data file.
folder_name=uigetdir('Raw Data','Select the raw data of the field');
folder_name_wr=strrep(folder_name,'Raw Data','Processed Data');
ind=regexp(folder_name_wr,'\');
fname_mat=fullfile([folder_name_wr,'.mat']);
if ~exist(fname_mat)
    disp('Making Raw Data File');
    if ~exist(folder_name_wr(1:ind(end)-1))
        
        mkdir(folder_name_wr(1:ind(end)-1))
    end
    fnames=dir(fullfile(folder_name,'*\*_Ch2_*.ome.tif*'));
    for fn=1:length(fnames)
        data(:,:,fn)=imread( fullfile(fnames(fn).folder,fnames(fn).name));
        if mod(fn,100)==0
            disp(fn)
        end
    end
    save (fname_mat, '-v7.3');
else
    disp ('Raw Data file already exists.. So loading from the file');
    load (fname_mat)
end
