%nf_extract v12 - takes sub folders containing groups and produces a single
%table with all intrinsic neuron properties

%additionally creates an array storing non scalar spike train data

%also creates a cell array with the original raw data for plotting traces
% added calculation of AP threshold for every single spike - so that you
% can look at AP threshold as a function of # of spike in train (similar to
% spike height)
%v5 added current injection measurements for IF curve etc

%this version 1_2 - as reported in Mattis et al 2022, uses the 1st AP at 2x
%rheobase for single AP measurements (lines 277-303)
function [] = nfex12_joanna(save_name)



files = dir;
directoryNames = {files([files.isdir]).name};
directoryNames = directoryNames(~ismember(directoryNames,{'.','..'}));
    


filenames=cell(0);


output_labels(1,:) = ["Recording_membrane_potential","Holding_Current","Calc_Vm","Input_Resistance","Membrane_time_constant","Rheobase",...
    "Inflection_rate","Max_rise_slope","ap_thresh_1","ap_thresh_2","ap_thresh_3","ap_amplitude","ap_peak",...
    "ap_rise_time","ap_10_90","ap_halfwidth","ahp1_amp","ahp1_time","ahp2_amp","ahp2_time","max_steady_state_firing",...
    "sag","max_instantaneous", "Sustained_Firing", "Burst_length", "Burst_APs", "Rheo_APs", "ISI_CoV", "Filename", "Group"];
output_labels = cellstr(output_labels);
data = table();
z=1;

for k = 1 : size(directoryNames,2)
    
    oldFolder = cd(char(directoryNames(k)));
    fileList = dir('*.abf');
    numfiles = size(fileList,1);
    

% data1=cell(numfiles, size(output_labels,2));
% data1 = cell2table(data1);

filenames1=struct2cell(fileList);  
filenames1=filenames1(1,:)';
labels1 = filenames1;
[labels1{1:size(filenames1,1)}] = deal(cell2mat(directoryNames(1,k)));
cells1=cell(numfiles,6);
for m = 1:numfiles
% Bring in ABF File as a 3D matrix
[d,si,Ts,Ft,l,w,h,matrix_exc,matrix_res,AP_threshold,filename] = getfile(0,fileList,m); %change AP threshold here
%defines size of excitation and response matrices, sampling rate/interval,
%and gives the data itself

%% passive properties and rheobase
[Recording_membrane_potential,holding_current,Calc_Vm,Input_resistancem,Membrane_time_constmean,act_pot,Rheobase,mem_pot_sag,duration,inj_window] = ...
    passive_prop_rheo(matrix_exc,matrix_res,AP_threshold,h,Ts);
%calculates most passive properties and finds rheobase, also defines
%current injection period - duration


%% Active properties

[infl_rate,max_dvdt,AP_thresh1,AP_thresh2,AP_thresh3,AP_peak,AP_height,AP_rise_time,AP_10_90,AP_hw,...
    ahp_t_s,ahp_a_s,ahp_t_l,ahp_a_l] = active_prop(matrix_res,AP_threshold,Ts,Ft,si,act_pot,inj_window);


%% spike counter

[ap_pks_matrix,ap_times_matrix,ISI_matrix,spike_count,sustained_firing,all_ap_locs_matrix] = ...
    spike_detector(h,Ft,Ts,matrix_res,AP_threshold,inj_window);

%% repetative firing properties

[ISI_CoV,ISI_Cov_mean,sfa,sfa_e,maximal_steady_state_FF,Maximal_inst_FF,freq_train,max_burst,Burst_length,Burst_APs,Rheo_Burst_APs] = ...
    repetative_firing_properties(ISI_matrix,spike_count,duration,Ts,h,ap_times_matrix,act_pot);

%%  AP threshold train

[AP_thresh_train] = APth_train(matrix_res,Ts,all_ap_locs_matrix);

[IC_steps] = get_current(matrix_exc);

%% output
% output(1,:) = ["Resting membrane potential","Input_Resistance","Membrane_time_constant","Rheobase",...
%     "Inflection_rate","Max_rise_slope","ap_thresh_1","ap_thresh_2","ap_thresh_3","ap_amplitude","ap_peak",...
%     "ap_rise_time","ap_10_90","ap_halfwidth","ahp1_amp","ahp1_time","ahp2_amp","ahp2_time","max_steady_state_firing",...
%     "sag","max_instantaneous", "Sustained Firing", "Burst length", "Burst APs", "Rheo APs", "ISI CoV"];

output(m,:) = [Recording_membrane_potential holding_current Calc_Vm Input_resistancem Membrane_time_constmean Rheobase ...
    infl_rate max_dvdt AP_thresh1 AP_thresh2 AP_thresh3 AP_height AP_peak...
    AP_rise_time AP_10_90 AP_hw ahp_a_s ahp_t_s ahp_a_l ahp_t_l maximal_steady_state_FF...
    mem_pot_sag Maximal_inst_FF max(sustained_firing) max_burst(1) max_burst(2) Rheo_Burst_APs ISI_Cov_mean];


    train.aptimes{z,1} = ap_times_matrix;
    train.appeaks{z,1} = ap_pks_matrix;
   
    train.sustained{z,1} = sustained_firing;

    train.spike{z,1} = spike_count;

    train.ISI_CoV{z,1} = ISI_CoV;

    train.sfa{z,1} = sfa;


    train.sfa_e{z,1} = sfa_e;



    train.Burst_l{z,1} = Burst_length;


    train.Burst_n{z,1} = Burst_APs;
    train.AP_thresh{z,1} = AP_thresh_train;
    train.IC_steps{z,1} = IC_steps;
    train.file(z,1) = cellstr(filenames1(m));
    train.group(z,1) = cellstr(directoryNames(k));
    raw_traces.exc{z,1} = matrix_exc;
    raw_traces.res{z,1} = matrix_res;
    raw_traces.Ts{z,1} = Ts;
    z=z+1;

end
data1 = array2table(output);
clear output
data1.filenames = filenames1;
data1.group = cellstr(repmat(directoryNames(k),numfiles,1));
data = [data;data1];

cd(oldFolder)

end
data.Properties.VariableNames = output_labels;

save(save_name, 'data', 'train', 'raw_traces');

end

%% 
%% 
%% functions


function [d,si,Ts,Ft,l,w,h,matrix_exc,matrix_res,AP_threshold,filename] = getfile(thresh,fileList,m)
switch nargin
        case 3
            filename = fileList(m).name;
        case 2
            filename = fileList;
        case 1
            filename = uigetfile('*.*');
        otherwise
            filename = 0;
end

[d,si,~] = abfload(filename);
filename = convertCharsToStrings(filename);
%AP_threshold = input(sprintf('\nWhat action potential cutoff would you like (mV)? ')); 
AP_threshold = thresh;
%Sampling time and rate

Ts = si * 1e-6;

Ft = 1/Ts;

% Gather dimensions of 3D matrix
[l,w,h] = size(d);

% Store 3D matrix in 2D matrices
if (w==2)

matrix_exc = zeros(l,h);
matrix_res = matrix_exc;
    
    for i = 1:h

        matrix_exc(:,i) = d(:,2,i);   % Excitation matrix (Current Injection)
        matrix_res(:,i) = d(:,1,i);   % Response matrix (Action Potentials)
    
    end

else
matrix_exc = zeros(l,h);
matrix_res = matrix_exc;
    
    for i = 1:h

        matrix_res(:,i) = d(:,1,i);   % Response matrix (Action Potentials)
    
    end
    
    disp(sprintf('\n\n\n***** NO Excitation curves!!!!! *******\n\n\n'))
    

     bbb = matrix_exc(:,1);
     bbb(12347:72347) = bbb(12347:72347) +1;
     bbb=logical(bbb);
for ii = 1:h
    matrix_exc(bbb,ii) = matrix_exc(bbb,ii) + (-100 + 25*(ii-1));
    
end


end

end


function [Recording_membrane_potential,holding_current,Calc_Vm,Input_resistancem,Membrane_time_constmean,act_pot,Rheobase,mem_pot_sag,duration,inj_window] = ...
    passive_prop_rheo(matrix_exc,matrix_res,AP_threshold,h,Ts)

%%Resting membrane potential



[value,index] = min(abs(mean(matrix_exc(:,:))-mean(matrix_exc(1:100,:))));
 if index == 1
     index = 2;%catches when there is no 0 current injection
 end
 
baseline = mean(matrix_exc(:,index));

Recording_membrane_potential = mean(matrix_res(:,index));  
holding_current = mean(matrix_exc(:,index));

% try
% disp(sprintf('\nResting membrane potential: %0.3f mV\n', Resting_membrane_potential))
% catch
% disp(sprintf('\nResting membrane potential is unknown. No sweep has 0 pA injection.\n'))
% end
% figure;plot(matrix_res)
% figure;plot(matrix_exc)
%% Input Resistance Delta V / Delta I

% Finds a negative excitation the first?

% for i = 1:h
%     
%     if(mean(matrix_exc(:,i))<baseline)                   % Finds excitation that is averag less
%                                                          % than zero or a negative excitation   
%         neg_inj = i;                                     % Stores the negative excitation index       
%         break                                           % Saves time
%     
%     end
%     
% end
neg_inj = 1; %assumes first sweep is hyperpolarizing
neg_injm = [neg_inj:index-1];  %creates array of neg injection indices
%if mean(matrix_exc(:,neg_inj)) < -20
    boolean = matrix_exc(:,neg_inj)<mean(matrix_exc(1:100,neg_inj))-20;                   % Finds points where current is being negatively injected
    boolean = matrix_exc(:,end)>mean(matrix_exc(1:100,end))+6; 
    %else
    %boolean = matrix_exc(:,neg_inj)<baseline-2;     %fixes issue if IC was recorded with wrong scale factor (ie values for pA very small)
    %matrix_exc = matrix_exc*19.7417;
    %end
inj_window=boolean;
a = find(boolean,1);
b = find(boolean,1,'last');
c = round((a+b)/2);
% figure;
 %plot(matrix_exc(boolean,neg_injm))

for i = 1:length(neg_injm)
    Del_I(i) = abs(mean(matrix_exc(1:1000,i)) - mean(matrix_exc(c:b,i)));          % Calculates Change in Current
    Del_V(i) = abs(mean(matrix_res(1:1000,i)) - mean(matrix_res(c:b,i))); % Calculates Change in voltage
    Input_resistance(i) = (Del_V(i)/Del_I(i))*1000;                                      % Displays input resitance
end
Input_resistancem= mean(Input_resistance);
Calc_Vm = Recording_membrane_potential - (holding_current*Input_resistancem)/1000;
% disp(sprintf('Input resistance: %0.3f MOhm\n', Input_resistancem))

%% Membrane time constant, Rheobase, Action Potential Amplitude, AP Threshold, Rise Time, AP half-width

for i = 1:h
    if size(matrix_res,2)>i 
    if(sum(matrix_res(boolean,i+1)>AP_threshold)>0)
        if(sum(matrix_res(boolean,i)>AP_threshold)>0)                          % Finds response with first nonspurious action potential
        
            if length(neg_injm)>1
            act_pot = (i-length(neg_injm)-1)*2+length(neg_injm)+1;   
            else
            act_pot = i;                                      % Stores the action potential index   
            end
        break                                             
    
        end
    end
    else % allows for sweeps with only a single trace with APs 
        if(sum(matrix_res(boolean,i)>AP_threshold)>0)   
        act_pot = i;
        break
        end
    end
    
end

if act_pot > size(matrix_res,2) %catches few cases where max current is < 2x rheobase
    act_pot = size(matrix_res,2);
end

%% fix for if there is spontaneous activity driving firing

%act_pot=act_pot+1;
%use if there is a sporadic firing eg from spontaneous activity

%% Membrane time constant

dur_boolean = boolean;

duration = sum(boolean);                                 % Finds duration current is active (# of points)

for i = 2:length(dur_boolean)
   
    if dur_boolean(i) ~= dur_boolean(i-1)
        start_pt = i;                                    % Finds the point at which current injection actually begins
        break
    end
end

fit_dur = 700;

%x = (1:fit_dur).*Ts*1000;                                % Time vector for fit

start_pt_buffer = start_pt + 50;                            %start time buffer
for i = 1:length(neg_injm)
y = matrix_res(start_pt_buffer:start_pt_buffer+fit_dur-1,i); % Creates vector to fit

x = (1:fit_dur).*Ts*1000;
gof.sse = -1;
    for j = 1:h

    if (mean(matrix_exc(1:start_pt,j))-mean(matrix_exc(start_pt_buffer:start_pt_buffer+fit_dur,j))<0) 
        if gof.sse < 0 || gof.sse > 30
        [f,gof] = fit(x',y,'exp2');                        % Fits membrane constant for selected negative injection traces

            Membrane_time_const(i) = 1/(-f.d);              % Membrane constant
            placeholder(i)=gof.sse;
        end
    end
    end
end
error1 = placeholder(:) < 5;
error2 = Membrane_time_const(:) >= 0;
error3 = Membrane_time_const(:) <= 50;
errort = error2 & error3 & error1;
errort2 = error2 & error3;
if mean(Membrane_time_const(errort)) > 0
    Membrane_time_constmean = mean(Membrane_time_const(errort));   %kicks out any values with SSE >5
else
    Membrane_time_constmean = mean(Membrane_time_const(errort2));
end
% disp(sprintf('Membrane time constant: %0.3f ms\n', Membrane_time_const))
% disp(sprintf('Membrane time constant mean: %0.3f ms\n', Membrane_time_constmean))

%% Rheobase

current_min = 1e8;

for i = 1:h                           % Finds DC Offset
            
    if(abs(mean(matrix_exc(dur_boolean,i))-mean(matrix_exc(~dur_boolean,i))) < current_min)              
        
        current_min = abs(mean(matrix_exc(dur_boolean,i))-mean(matrix_exc(~dur_boolean,i)));
    
        offset_index = i;
        
    end
    
end


Rheobase = mean(matrix_exc(dur_boolean,act_pot)) - mean(matrix_exc(~dur_boolean,act_pot));%mean(matrix_exc(:,offset_index));  % Calcualtes rheobase -fixed to take from the same sweep

% disp(sprintf('Rheobase: %0.3f pA\n', Rheobase))


%% Membrane Potential Sag - doesn't seem correct. looks like it calculates the mean of non current injection to the mean of current injection

%for i = 1:h

    %if (mean(matrix_exc(:,i))<0) 
        
        %mem_pot_sag = mean(matrix_res(~dur_boolean,i))/mean(matrix_res(dur_boolean,i));  % Calculate sag for negative injection
        
        %break
  
    %end
    
%end

for i = 1:h             %corrected

    if (mean(matrix_exc(1:start_pt,j))-mean(matrix_exc(start_pt_buffer:start_pt_buffer+fit_dur,j))<0) 
        for j = 2:length(dur_boolean)
            if dur_boolean(j) - dur_boolean(j-1) == -1
                end_injection_index = j; %finds the end of the current injection
                break
            end
        end
        %meansagvals = matrix_res((end_injection_index-(.3/Ts)):end_injection_index,i);
        %meansag = mean(meansagvals);
       
        %sag
        break
  
    end
    
end

denominator = mean(matrix_res((1:1000),1)) - mean(matrix_res((end_injection_index-(.3/Ts)):end_injection_index,1));
mem_pot_sag = (min(matrix_res(dur_boolean,1))- mean(matrix_res((end_injection_index-(.3/Ts)):end_injection_index)))/denominator;  % Calculate sag for negative injection
        %uses the mean for the last 300ms of current injection to calculate
        
% disp(sprintf('\nMembrane Potential Sag: %0.3f \n', mem_pot_sag))
end


function [infl_rate,max_dvdt,AP_thresh1,AP_thresh2,AP_thresh3,AP_peak,AP_height,AP_rise_time,AP_10_90,AP_hw,...
    ahp_t_s,ahp_a_s,ahp_t_l,ahp_a_l] = active_prop(matrix_res,AP_threshold,Ts,Ft,si,act_pot,inj_window)

%% Find first AP at Rheobase position

 boolean = matrix_res(:,act_pot)>=AP_threshold;
 first_sweep = matrix_res(:,act_pot);
 first_sweep(first_sweep<AP_threshold)=AP_threshold;first_aps = inj_window.*first_sweep;            %all aps in rheobase sweep - fixed 1_4
 [ap_pks,ap_locs] = findpeaks(first_aps,'MinPeakDistance',100);  %finds peaks and indices %changed to 100 from 40
  
 index6 = min(ap_locs);     %takes first peak index
 
%[peak_value,index6] = max(matrix_res(:,act_pot)); only finds max, gets
%confused if there is more than one AP at rheobase -eg bursting 

startpt = index6 - .0025*Ft; % changed to 2.5 ms for slower APs 

endpt = index6 + .0015*Ft;

%% Calculate Derivative 

Fir_Deriv = (matrix_res(startpt+1:endpt,act_pot)-matrix_res(startpt:endpt-1,act_pot))./(si*1e-3);
Sec_Deriv = (Fir_Deriv(2:end)-Fir_Deriv(1:end-1))./(si*1e-3);

%% Inflection rate

 factor = 10;

 interp_TS = interp1(1:length(startpt:endpt),matrix_res(startpt:endpt,act_pot),1:1/factor:length(startpt:endpt));

[peak_value,peak_index] = max(interp_TS);

interp_deriv = interp1(1:length(Fir_Deriv),Fir_Deriv,1:1/factor:length(Fir_Deriv));

interp_sec_deriv = interp1(1:length(Sec_Deriv),Sec_Deriv,1:1/factor:length(Sec_Deriv));

% Inflection rate

[pk_value_interp, interp_pk_index] = max(interp_TS);

Diff = abs(interp_deriv(1:interp_pk_index-20) - 10); %peak index - 10 was not sufficient

[value,index7] = min(Diff);
if max(interp_deriv)<10
    Diff = abs(interp_deriv(1:interp_pk_index-20) - 5); %uses 5mv/s if ap is too slow

[value,index7] = min(Diff);
end

if index7 < factor+1
index7 = factor+1;
end

deriv_steps = interp_deriv(index7-1*factor:index7+1*factor);

TS_steps = interp_TS(index7-1*factor:index7+1*factor);

P = polyfit(TS_steps,deriv_steps,1);
infl_rate = abs(P(1));



%% Action Potential Threshold and peak of dV/dt

Derivative = matrix_res(2:end,act_pot) - matrix_res(1:end-1,act_pot); % Takes derivative of response

sec_Derivative = Derivative(2:end)-Derivative(1:end-1);               % Takes second derivative of response


%this does not appear to be used except in the half width calculation,
%which would occasionally pick up the wrong points if there were very fast
%transients from noise etc
%[pks,locs] = findpeaks(sec_Derivative);
%matrix = [pks,locs];                           
%matrix = sortrows(matrix);

%if (matrix(end,2)<matrix(end-1,2))

   % index = matrix(end,2);
%else
   % index = matrix(end-1,2);
%end                        
                                   % Finds the max position of action potential inflection point

AP_thresh1 = interp_TS(index7);               % Displays Action potential threshold

% Change_rate = Derivative./(si*1e-3);
max_dvdt = max(Derivative./(si*1e-3));
[sec_der_max,index9] = max(interp_sec_deriv);
s_sec_dif = round(index9 - (.001*factor/Ts));
if s_sec_dif <= 0
    s_sec_dif = 1;
end

sec_deriv_firsthalf = interp_sec_deriv((s_sec_dif+1):index9);
Diff2 = abs(sec_deriv_firsthalf - (sec_der_max*.05));
%Diff2 = abs(interp_sec_deriv - (sec_der_max*.05)); this doesnt work
%because the second derivative will cross the same point twice. this can
%pick up the second one

sec_dif_l = length(Diff2);
[val,index10a] = min(Diff2);
index10 = index10a + s_sec_dif;
AP_thresh2 = interp_TS(index10);
AP_thresh3 = interp_TS(index9);
% Display results

% disp(sprintf('Maximum dV/dt: %0.3f mV/ms\n', max(Change_rate)))
% 
% disp(sprintf('Action potential threshold: %0.3f mV (Method 1)\n', interp_TS(index7)))
% 
% disp(sprintf('Action potential threshold: %0.3f mV (Method 2)\n', interp_TS(index10)))
% 
% disp(sprintf('Action potential threshold: %0.3f mV (Max 2nd Derivative)\n', interp_TS(index9)))

% figure;
% plot(Derivative)
% hold all
% plot(sec_Derivative)
% plot(Change_rate)
% plot(matrix_res(1:end,act_pot))

%% Action Potential height/amplitude

AP_peak = matrix_res(index6,act_pot);      %  peak value and index
                                       
AP_height = AP_peak - interp_TS(index7);  % Gives AP amplitude

% disp(sprintf('Action potential height/amplitude: %0.3f mV\n', AP_height))
% 
% disp(sprintf('Action potential peak voltage: %0.3f mV\n', AP_peak))

%% Action Potential rise time Rise Time

Action_potential_rise_time = ((peak_index-index7)/factor)*Ts;                               % AP rise time
AP_rise_time = Action_potential_rise_time*1000; %convert to ms
%disp(sprintf('Action potential rise time: %0.3f ms\n', Action_potential_rise_time*1000))

%% 90/10 ratio

[~,peak_index2] = max(interp_TS);

ten_vol = (AP_height*.1) + interp_TS(index7);

ninety_vol = (AP_height*.9) + interp_TS(index7);

[~,ten_index] = min(abs(interp_TS(1:peak_index2) - ten_vol));

[~,ninety_index] = min(abs(interp_TS(1:peak_index2) - ninety_vol));

AP_10_90 = ((ninety_index-ten_index)/factor)*Ts*1000;
%disp(sprintf('10-90 rise time: %0.3f ms\n', ((ninety_index-ten_index)/factor)*Ts*1000))



%% Action Potential Half Width
Half_width = (AP_height/2)+AP_thresh1;     % Finds halfwidth voltage

scale = 100;

xq = 1:1/scale:100-1/scale;

vq = interp1(matrix_res(index6-40:index6+40,act_pot),xq);  

[peak,index_peak] = max(vq);

[value,index3] = min(abs(vq(1:index_peak)-Half_width)); % Finds nearest point to HW voltage left of peak

[value,index4] = min(abs(vq(index_peak:length(vq))-Half_width)); % Finds nearest voltage to HW voltage right of peak

Action_potential_half_width = abs(index3 - (index4 + index_peak) );       % Calculates time based on point separation
AP_hw = Action_potential_half_width*Ts*1000/scale; %in ms
%disp(sprintf('Action potential half width: %0.3f ms\n', Action_potential_half_width*Ts*1000/scale))

% figure;
% plot(xq,vq,'*')
% hold all
% plot(matrix_res(index:index+100,act_pot),'o')
% plot(index3/scale+1,vq(index3),'*')
% plot((index4+index_peak)/scale+1,vq(index4+index_peak),'*')
%% Action potential after-hyperpolarization amplitude

threshold_window = .002/Ts; %2ms window
ahp_short = .005/Ts; %5ms window
ahp_long = .1/Ts; %100ms window
ap_thresh_window = matrix_res((index6 - threshold_window):index6,act_pot);
[val1,index11] = min(abs(AP_thresh1 - ap_thresh_window));       %finds AP in non interpolated value and index

non_interp_index = index6-(length(ap_thresh_window) - index11);
non_interp_ap = matrix_res(non_interp_index,act_pot);

[AP_AHP_short,index12] = min(matrix_res(index6:(index6+ahp_short),act_pot));  %finds minimum in 5ms window
AP_AHP_time_short = index12 * Ts;

[AP_AHP_long,index13] = min(matrix_res(index6+ahp_short:(index6+ahp_long),act_pot)); %finds minimum in 100ms window
AP_AHP_time_long = (index13 + ahp_short)* Ts;

% disp(sprintf('Short Action potential after-hyperpolarization amplitude: %0.3f mV\n', (AP_AHP_short-non_interp_ap)))
% disp(sprintf('Long Action potential after-hyperpolarization amplitude: %0.3f mV\n', (AP_AHP_long-non_interp_ap)))
% disp(sprintf('Short Action potential after-hyperpolarization time: %0.3f ms\n', AP_AHP_time_short*1000))
% disp(sprintf('Long Action potential after-hyperpolarization time: %0.3f ms\n', AP_AHP_time_long*1000))

ahp_t_s = AP_AHP_time_short*1000;
ahp_a_s = AP_AHP_short-non_interp_ap;
ahp_t_l = AP_AHP_time_long*1000;
ahp_a_l = AP_AHP_long-non_interp_ap;
end

function [ap_pks_matrix,ap_times_matrix,ISI_matrix,spike_count,sustained_firing,all_ap_locs_matrix] = ...
    spike_detector(h,Ft,Ts,matrix_res,AP_threshold,inj_window)
%% Spike Counter and maximal steady-state firing frequnecy and Spike-frequency adaptations

spike_count = zeros(1,h);

for i = 1:h
    k = 1;
   if (sum(matrix_res(:,i).*inj_window>=AP_threshold)>0)                        % Does this response contain action potentials? nfex10 added specificity to only during current injection
    
    boolean = matrix_res(:,i).*inj_window>=AP_threshold;                        
    bob = boolean;
    for j = 2:length(boolean)
               
        if boolean(j) ~= boolean(j-1)                   % Find action potential start/end
            
            spike_count(i) = spike_count(i) + .5;       % Adds 1/2 to count for every start/end
            
        end
    end
   end
end

ap_pks_matrix = zeros(max(spike_count),size(matrix_res,2));
all_ap_locs_matrix = zeros(max(spike_count),size(matrix_res,2));
for i = 1:size(matrix_res,2)
%boolean = matrix_res(:,i).*inj_window>=AP_threshold;
 peak_sweep = matrix_res(:,i).*inj_window;
 all_aps = peak_sweep;all_aps(all_aps<AP_threshold)=AP_threshold;           %all aps in each sweep - fixed 1.4

 [all_ap_pks,all_ap_locs] = findpeaks(all_aps,'MinPeakDistance',.003*Ft,'MinPeakHeight',AP_threshold,'MinPeakProminence',4);  %finds peaks and indices capped at 500hz
 
 for j = 1:length(all_ap_pks)
     ap_pks_matrix(j,i) = all_ap_pks(j);
     all_ap_locs_matrix(j,i) = all_ap_locs(j);
 end
 ap_pks_matrix(j+1,i) = ap_pks_matrix(j,i);
  %all_ap_locs_matrix(j+1,i) = .7*Ft;    %adds final peak at end of sweep -
  %use if you want to measure ISI COV with endpoint
end
ap_times_matrix = all_ap_locs_matrix*Ts; %ap times in s
sustained_firing = (sum(ap_times_matrix>0.4,1)/0.3)';%changed to 300ms
for i = 1:size(ap_times_matrix,2)
    for j = 2:size(ap_times_matrix,1)
        ISI_matrix(j,i) = ap_times_matrix(j,i)-ap_times_matrix(j-1,i);
    end
end
ap_times_matrix(ap_times_matrix==0) = NaN;
ISI_matrix(ISI_matrix==0) = NaN;  
ISI_matrix(ISI_matrix<0) = NaN; 
spike_count = spike_count';
end

function [ISI_CoV,ISI_Cov_mean,sfa,sfa_e,maximal_steady_state_FF,Maximal_inst_FF,freq_train,max_burst,Burst_length,Burst_APs,Rheo_Burst_APs] = ...
    repetative_firing_properties(ISI_matrix,spike_count,duration,Ts,h,ap_times_matrix,act_pot)

%% ISI coefficient of variation per sweep and mean
%ratio of standard deviation to mean for each sweep, then averaged across
%sweeps
for  i = 1:size(ISI_matrix,2)
    ISI_CoV(i,1) = std(ISI_matrix(:,i),'omitnan')/mean(ISI_matrix(:,i),'omitnan');
end

ISI_Cov_mean = mean(ISI_CoV,'omitnan');
% disp(sprintf('ISI CoV: %0.3f \n', ISI_Cov_mean))


%% Spike frequency adaptation

for i = 1:h

    if spike_count(i)>10
        sfa(i) = ISI_matrix(2,i)/ISI_matrix(11,i);                        % Spike frequency adaptation (ISI_10/ISI_1)
    elseif spike_count(i)>3
        sfa(i) = ISI_matrix(2,i)/ISI_matrix(3,i);                         % Spike frequency adaptation (ISI_2/ISI_1)
    else 
        sfa(i) = NaN; %not enough APs for SFA
    end
    
end

for i = 1:h
   
    if spike_count(i)>1
        sfa_e(i) = ISI_matrix(2,i)/ISI_matrix(spike_count(i),i);                        % Spike frequency adaptation (ISI_last/ISI_1), excludes the last fake spike
    else
        sfa_e(i) = NaN;                         % Spike frequency adaptation (ISI_2/ISI_1)
    
   end
end


%% Maximal instantaneous firing frequency

[value,index5] = max(spike_count);                      % Finds max number of spikes

Maximal_inst_FF = (1/min(ISI_matrix(:,index5)));    % Calculated based on smallest ISI

% disp(sprintf('Maximal instantaneous firing frequency: %0.3f Hz\n', Maximal_inst_FF))
% Maximal Steady State Firing Frequency
maximal_steady_state_FF = max(spike_count)*(1/(duration*Ts));


%% Instantaneous frequency plots

% instantaneous frequency function of time in sweep

% for i = 1:size(ap_times_matrix,2)
%     for j = 1:size(IS_interval_matrix,1)
%     IF_sweep(j,2*i-1) = ap_times_matrix(j,i)*1000;%time in ms
%     IF_sweep(j,2*i) = 1/IS_interval_matrix(j,i);%freq at that time
% end
% end
clear freq_matrix
clear freq_sweep
clear previous
freq_matrix = 1./ISI_matrix;

for i = 1 : size(freq_matrix,2)
    if spike_count(i)>1
        temp = freq_matrix(:,i);
    temp2 = max(find(temp > 0));
    if freq_matrix(temp2,i) > freq_matrix(temp2-1,i)
        freq_matrix(temp2,i) = freq_matrix(temp2-1,i);
    end
    end
    
end
        
    
   
  
start_current = .1;
end_current = .7;  %start and end times
bin = .01; %sample bin 10ms
bins = (end_current - start_current)/bin + 1;

k = 1;
for j = start_current : bin : end_current
   
%     boolean = IF_sweep(:,2*i-1) > j & IF_sweep(:,2*i-1) <= j + bins;
    boolean  = ap_times_matrix > j - bin & ap_times_matrix <= j;
%     boolean = boolean(1:size(boolean,1)-1,:);
    freq_sweep = freq_matrix .* boolean; 
    freq_sweep(freq_sweep==0)=NaN;
    
    for i = 1:size(ap_times_matrix,2)
        freq_train(k,i) = mean(freq_sweep(:,i),'omitnan');
        if isnan(freq_train(k,i)) & k > min(find(freq_train(:,i)>0))
            previous = find(freq_matrix(:,i)>0);
            freq_train(k,i) = freq_matrix(max(previous),i);
        end
        
    end
k = k +1;

end

%% burst length and APs per burst - wasnt quite right, updated below
% Burst_length = zeros(size(freq_train,2),1);
% Burst_APs = zeros(size(freq_train,2),1);
% for i = 1 : size(freq_train,2)
%     for j = 2:size(freq_train,1)
%         if freq_train(j,i) < 50 && freq_train(j-1,i) > 50
%             Burst_length(i,1) = bin*(j-1);
%             Burst_APs(i,1) = sum(ap_times_matrix(:,i)<j*bin+.1);
%             break
%         end
%             
%         if j == size(freq_train,1) && freq_train(j-1,i) > 50 
%             Burst_length(i,1) = bin*(j-1);
%             Burst_APs(i,1) = sum(ap_times_matrix(:,i)<j*bin+.1);
%         end
%     end
% end

Burst_length = zeros(size(freq_matrix,2),1);
Burst_APs = zeros(size(freq_matrix,2),1);
for i = 1 : size(freq_matrix,2)
    [~,lastap]=max(ap_times_matrix(ap_times_matrix(:,i)>0,i));
    if isempty(lastap)
        lastap=0;
    end
    for j = 2:size(freq_matrix,1)
        if freq_matrix(j,i) < 50 && freq_matrix(j-1,i) > 50 || j == lastap && freq_matrix(j-1,i) > 50
            Burst_length(i,1) = ap_times_matrix(j-1,i) - ap_times_matrix(1,i);
            Burst_APs(i,1) = j-1;
            break
        else
        end
    end
end

max_burst = [max(Burst_length) max(Burst_APs)]; % use the max burst to characterize vs. using standard 3X rheobase to characterize


% mRheo =3;
% plotstep = round((mRheo-1)*(act_pot-6)+act_pot);%assumes starting at -50pa and 10pa steps
%  if plotstep > size(Burst_length,1)
%     plotstep = size(Burst_length,1);
%  end
%     
% max_burst = [Burst_length(plotstep) Burst_APs(plotstep)];

Rheo_Burst_APs = 1;
% for j = 2:size(freq_matrix,1)
%         if freq_matrix(j,act_pot) < 100 && freq_matrix(j-1,act_pot) > 100% needs fake spike to work 
%             Rheo_Burst_APs = j-1;
%             break
%         else
%         end
%    end
indx = freq_matrix(:,act_pot) > 100;
indx(1)=1;
indx = find(indx);

if max(indx)>1
    Rheo_Burst_APs = max(indx);
end

end


function [AP_thresh_train] = APth_train(matrix_res,Ts,all_ap_locs_matrix)




all_ap_locs_matrix(all_ap_locs_matrix==35000)=0;%gets rid of artificial spike at end
ind = find(all_ap_locs_matrix);
[jj,ii] = find(all_ap_locs_matrix);
ap_locs = all_ap_locs_matrix(ind);
AP_thresh_train=zeros(size(all_ap_locs_matrix));
sweeps = zeros(size(ind,1),round(0.001/Ts))';
for j = 1:length(ind)
sweeps(:,j) = matrix_res(ap_locs(j)-round(0.001/Ts)+1:ap_locs(j),ii(j));
end

sweepd = (sweeps(2:end,:)-sweeps(1:end-1,:))./(Ts*1000);
sweepdd = (sweepd(2:end,:)-sweepd(1:end-1,:))./(Ts*1000);



thr_ind = abs(sweepd(1:end-10,:) - 10); %peak index - 10 ensures first crossing of 10mv/ms

[~,ap_ind] = min(thr_ind);
for i = 1:size(sweeps,2)
AP_thresh_train(jj(i),ii(i)) = sweeps(ap_ind(i),i);
end

end

function [IC_steps] = get_current(matrix_exc)

baseline = mean(matrix_exc(1:100,end));
boolean = matrix_exc(:,end)>baseline+8;                   % Finds points where current is being negatively injected in first sweep
a = find(boolean,1);
b = find(boolean,1,'last');
c = round((a+b)/2);


for i = 1:size(matrix_exc,2)
    IC_steps(i) = round(mean(matrix_exc(c:b,i)-mean(matrix_exc(1:100,i))),-1);          % Calculates Change in Current
end

end
