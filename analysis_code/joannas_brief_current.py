import os
import sys 
sys.dont_write_bytecode = True
from pyabf import ABF
from .analyze_abf import CurrentStepsData

def analyze_bc(ABF_LOCATION, ANALYSIS_OUTPUT_FILE):
    if os.path.isdir(ABF_LOCATION):
        abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
    else:
        abf_files = [ABF_LOCATION]

    # Print the files we're analyzing as a sanity check
    print('Analyzing the following files:\n{}'.format(abf_files))

    # Gathering data from the abf files
    ahp_output = {}
    ahp_time_output = {}

    for filepath in abf_files:
        abf = ABF(filepath)
        experiment = CurrentStepsData(abf)

        filename = os.path.basename(filepath)
        print('Analyzing {}'.format(filename))

        print('{} contains {} sweeps'.format(filename, len(experiment.sweeps)))

        ahp_output[filename], ahp_time_output[filename] = experiment.get_ahp_amplitude_and_time()


    # Writing the additional analysis to output file
    with open(ANALYSIS_OUTPUT_FILE, 'w') as f:
        f.write("filename,AHP Amplitude (mV),AHP Time (ms)\n")
        for filename in ahp_output:
            f.write('{},{},{}\n'.format(
                filename,
                ahp_output[filename],
                ahp_time_output[filename]
            ))
