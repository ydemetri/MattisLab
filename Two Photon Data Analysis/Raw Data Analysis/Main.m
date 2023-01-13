%% Analysis of Two Photon Data
%Raw data files from same field are taken as input.
%All the image files are combined to a single data file.
%The images are motion corrected
%The individual cells are identified
%The activity of each cell in the field are shown in plots and excel

clc;
clear all;
close all;

cur_dir=pwd;

%The toolboxes are added to the path
addpath(genpath('Toolboxes downloaded\CaImAn-MATLAB-master'));
addpath(genpath('Toolboxes downloaded\NoRMCorre-master'));
addpath(genpath('Toolboxes downloaded\NeuroSeg-master'));

%Analysis
Combine_Field_data;
Motion_correction;
Average_Frame;
NeuroSeg;
user=input('Please press y to continue to results: ','s');
if user=='y'
    cd (cur_dir)
    cond_list=1:size(M_rg,3)/500;
    pk_loc_mat=[];
    Data_Results_edit_bg_edit_window;
end