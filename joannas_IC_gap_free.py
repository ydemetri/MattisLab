import os
from pyabf import ABF
from analyze_abf import CurrentClampGapFreeData
import numpy as np

ABF_LOCATION = r'C:\Users\mattisj\Desktop\9-Patching\GC juvenile Scn1a\IC gap free'
IC_GAP_FREE_OUTPUT_FILE = r'C:\Users\mattisj\Desktop\9-Patching\GC juvenile Scn1a\IC gap free GC juvenile Scn1a.csv'

if os.path.isdir(ABF_LOCATION):
    abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
else:
    abf_files = [ABF_LOCATION]

# Print the files we're analyzing as a sanity check
print('Analyzing the following files:\n{}'.format(abf_files))

# Gathering data from the abf files
resting_potential_output = {}

for filepath in abf_files:
    abf = ABF(filepath)
    experiment = CurrentClampGapFreeData(abf)

    filename = os.path.basename(filepath)
    print('Analyzing {}'.format(filename))
    resting_potential_output[os.path.basename(filename)] = []
    resting_potential_output[filename] = experiment.get_resting_potential()

# Writing the additional analysis to output file
with open(IC_GAP_FREE_OUTPUT_FILE, 'w') as f:
    f.write("filename, resting potential\n")
    for filename in resting_potential_output:
        f.write('{}, {}\n'.format(filename, resting_potential_output[filename]))
