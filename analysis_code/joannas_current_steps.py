import os
import sys 
sys.dont_write_bytecode = True
from pyabf import ABF
from .analyze_abf import CurrentStepsData

def analyze_cc(ABF_LOCATION, CURRENT_VS_APS_OUTPUT_FILE, ANALYSIS_OUTPUT_FILE, SAG_OUTPUT_FILE):
    if os.path.isdir(ABF_LOCATION):
        abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
    else:
        abf_files = [ABF_LOCATION]

    # Print the files we're analyzing as a sanity check
    print('Analyzing the following files:\n{}'.format(abf_files))

    # Gathering data from the abf files
    current_vs_aps_output = {}
    ap_half_width_output = {}
    ap_peak_output = {}
    ap_amplitude_output = {}
    ap_rise_time_output = {}
    ap_threshold_1_output = {}
    rheobase_output = {}
    max_instantaneous_firing_frequency_output = {}
    max_steady_state_firing_frequency_output = {}
    spike_frequency_adaptation_10_output = {}
    spike_frequency_adaptation_N_output = {}
    time_constant_output = {}
    sag_per_current_output = {}

    for filepath in abf_files:
        abf = ABF(filepath)
        experiment = CurrentStepsData(abf)

        filename = os.path.basename(filepath)
        print('Analyzing {}'.format(filename))
        current_vs_aps_output[os.path.basename(filename)] = []

        print('{} contains {} sweeps'.format(filename, len(experiment.sweeps)))
        
        current_vs_aps_output[filename] = list(zip(
            experiment.get_current_step_sizes(), experiment.get_ap_counts()[1]
        ))

        sag = experiment.get_sag()
        sag_per_current_output[os.path.basename(filename)] = []
        sag_per_current_output[filename] = list(zip(sag[0], sag[1]))


        # individual AP characteristics
        ap_half_width_output[filename], ap_peak_output[filename] = experiment.get_ap_half_width_and_peak()
        ap_amplitude_output[filename] = experiment.get_ap_amplitude()
        ap_rise_time_output[filename] = experiment.get_ap_rise_time()
        ap_threshold_1_output[filename] = experiment.get_ap_threshold()

        #Characteristic of cell
        time_constant_output[filename] = experiment.get_time_constant()

        # characteristics of spike train
        rheobase_output[filename] = experiment.get_rheobase()
        max_instantaneous_firing_frequency_output[filename] = experiment.get_max_instantaneous_firing_frequency()
        max_steady_state_firing_frequency_output[filename] = experiment.get_max_steady_state_firing_frequency()
        spike_frequency_adaptation_10_output[filename], spike_frequency_adaptation_N_output[filename] = experiment.get_spike_frequency_adaptation()

    # Writing the % spikes per current step data to output file
    max_sweeps = len(max(current_vs_aps_output.values(), key=lambda x: len(x)))
    filenames = sorted(current_vs_aps_output.keys())
    print('max_sweeps is {}'.format(max_sweeps))
    with open(CURRENT_VS_APS_OUTPUT_FILE, 'w') as f:
        header = []
        index = 0
        for s in filenames:
            header.append(s)
            header.append("Values_{}".format(index))
            index += 1
        f.write(','.join(header))
        f.write('\n')

        for i in range(max_sweeps):
            for filename in filenames:
                try:
                    f.write('{},{},'.format(*current_vs_aps_output[filename][i]))
                except IndexError:
                    f.write(',,')
            f.write('\n')

    
    # Writing the sag data to output file
    max_sweeps = len(max(sag_per_current_output.values(), key=lambda x: len(x)))
    filenames = sorted(sag_per_current_output.keys())
    print('max_sweeps is {}'.format(max_sweeps))
    with open(SAG_OUTPUT_FILE, 'w') as f:
        f.write(','.join(['{}, '.format(s) for s in filenames]))
        f.write('\n')

        for i in range(max_sweeps):
            for filename in filenames:
                try:
                    f.write('{},{},'.format(*sag_per_current_output[filename][i]))
                except IndexError:
                    f.write(',,')
            f.write('\n')

    # Writing the additional analysis to output file
    with open(ANALYSIS_OUTPUT_FILE, 'w') as f:
        f.write("filename,AP Halfwidth (ms),AP Peak (mV),AP Amplitude (mV),AP Rise Time (ms),AP Threshold (mV),Rheobase (pA),Time Constant (ms),Max Instantaneous (Hz),Max Steady-state (Hz),SFA10,SFAn\n")
        for filename in ap_half_width_output:
            f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                filename,
                ap_half_width_output[filename],
                ap_peak_output[filename],
                ap_amplitude_output[filename],
                ap_rise_time_output[filename],
                ap_threshold_1_output[filename],
                rheobase_output[filename],
                time_constant_output[filename],
                max_instantaneous_firing_frequency_output[filename],
                max_steady_state_firing_frequency_output[filename],
                spike_frequency_adaptation_10_output[filename],
                spike_frequency_adaptation_N_output[filename]
            ))
