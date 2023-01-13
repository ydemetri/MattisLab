%The activity of each cell in the field are shown in plots and excel
if exist(fullfile([folder_name_wr,'average_Image_ROIdata.mat']))
    load (fullfile([folder_name_wr,'average_Image_ROIdata.mat']));
else
    load (fullfile([folder_name_wr,'average_selected_Image_ROIdata.mat']));
end
Interval=2;
Pixels=ParametersOutput.Pixels;
nFrames=size(M_rg,3);
Conditions=nFrames/500;
Cellnum=length(ParametersOutput.xypos);
F=zeros([nFrames Cellnum]);
vidHeight = size(M_rg,2);
vidWidth  = size(M_rg,1);
[bb,aa]=butter(3,[0.15]);
F=[];
for fi=1:nFrames
    M_rgc=M_rg(:,:,fi);
    for ci=1:Cellnum
        xi = Pixels{ci}(1,:);
        yi = Pixels{ci}(2,:);
        ind=sub2ind([vidHeight vidWidth],yi,xi);
        F(fi,ci)=mean(M_rgc(ind));
    end
    
end
F_filt=[];
for ci=1:Cellnum-1
    F_filt(:,ci)=filtfilt(bb,aa,F(:,ci));
end
F_reshaped=reshape(F_filt,[500,Conditions,Cellnum-1]);
F0=squeeze(mean(F_reshaped(5:50,:,:)));
y_labe=num2cell(0:5:Cellnum-1);
for cci=1:length(y_labe)
    y_labe{cci}=num2str(y_labe{cci});
end
dF_data=[];
dF_std=[];
dF_mean=[];
dF_data=[];
count=[];
val=[];
loc=[];
wid=[];
for cond=1:size(F_reshaped,2)
    figure;
    title(['Calcium traces for Condition' num2str(cond)])
    hold on;
    ymax=0;
    ymin=0;
    ind=find(cond_list==cond);%%%%%%included list if condition list is not in order
    for ii=1:Cellnum-1
        F_temp=(F_reshaped(:,ind,ii)-F0(ind,ii))/F0(ind,ii);
        if isempty(strfind(folder_name_wr,'DW'))
            dF_std(cond,ii)=std(F_temp(:,:));
            dF_mean(cond,ii)=mean(F_temp(:,:));
        else
            dF_std(cond,ii)=std(F_temp(400:end,:));
            dF_mean(cond,ii)=mean(F_temp(400:end,:));
        end
        F_temp(F_temp>100)=0;
        plot((1:size(F_reshaped,1)),F_temp+(ii)*Interval);
        dF_data(:,cond,ii)=F_temp;
        ymin = min(min(F_temp+(ii)*Interval),ymin);
        ymax = max(max(F_temp+(ii)*Interval),ymax);
        
        [v,l,w,p]=findpeaks(dF_data(:,cond,ii),'WidthReference','halfheight');
        if isempty(pk_loc_mat)
            Pk_loc=200;
        else
            Pk_loc=pk_loc_mat(ind);
        end
        loc_t=l(find(l>=Pk_loc,1,'first'));
        if (~isempty(loc_t) && loc_t>=Pk_loc && loc_t<=Pk_loc+15) && ((v(l==loc_t)-dF_mean(cond,ii))>3*dF_std(cond,ii))%50 300
            count(cond,ii)=1;
            val(cond,ii)=v(l==loc_t);
            loc(cond,ii)=loc_t;
            wid(cond,ii)=w(l==loc_t);
            hold on;plot(loc(cond,ii)-5,val(cond,ii)+(ii)*Interval+0.1,'k*','MarkerSize',4)
        else
            count(cond,ii)=0;
            val(cond,ii)=NaN;
            loc(cond,ii)=NaN;
            wid(cond,ii)=NaN;
        end
        xlabel('Frame Number');
        ylabel('Cell Number');
        xmin = 0;
        xmax = size(F_reshaped,1);
        ymin=0;
        axis([xmin xmax ymin ymax+Interval]);
        set(gca, 'YTick', [Interval Interval*5:Interval*5:Cellnum*Interval]);
        set(gca,'YTickLabel',y_labe);
        set(gca,'FontName','Times New Roman','FontSize',14);
        saveas(gca,fullfile([folder_name_wr,'_condition_' num2str(cond) '.tif']));
    end
    fname_xl=fullfile([folder_name_wr,'_results.xls']);
    xlswrite(fname_xl,{'Condition';'Activated Cells';'Total Cells';'Peak Amplitude';'Peak Width'},'Summary','A1');
    xlswrite(fname_xl,[1:Conditions],'Summary','B1');
    xlswrite(fname_xl,[sum(count,2)'],'Summary','B2');
    xlswrite(fname_xl,[repmat(Cellnum-1,1,Conditions)],'Summary','B3');
    xlswrite(fname_xl,[nanmean(val,2)'],'Summary','B4');
    xlswrite(fname_xl,nanmean(wid,2)','Summary','B5');
    xlswrite(fname_xl,{'Condition'},'Peak Amplitude','B1');
    xlswrite(fname_xl,{'Cell'},'Peak Amplitude','A2');
    xlswrite(fname_xl,[1:Conditions],'Peak Amplitude','B2');
    xlswrite(fname_xl,[0:Cellnum-2]','Peak Amplitude','A3');
    xlswrite(fname_xl,[val]','Peak Amplitude','B3');
    xlswrite(fname_xl,{'Condition'},'Activation','B1');
    xlswrite(fname_xl,{'Cell'},'Activation','A2');
    xlswrite(fname_xl,[1:Conditions],'Activation','B2');
    xlswrite(fname_xl,[0:Cellnum-2]','Activation','A3');
    xlswrite(fname_xl,[count]','Activation','B3');
end
