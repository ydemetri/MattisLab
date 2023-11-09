import sys 
sys.dont_write_bytecode = True

import tkinter as tk
from analysis_code import joannas_vc_test #get_input_resistance_from_vc
from analysis_code import joannas_IC_gap_free # get_resting_potential_from_gf
from analysis_code import joannas_current_steps # analyze_cc
from analysis_code import joannas_brief_current # analyze_bc
import pandas as pd
from scipy.stats import ttest_ind
import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import threading
from collections import defaultdict
import statsmodels.stats.multitest as multi

# f1 -> wt and f2 -> ex

path = "" # path to folders containing data
f1 = "WT" # name of folder containing data of group 1 in .abf
f2 = "EX" # name of folder containing data of group 2 in .abf
min_c = "-60" # minimum current in current steps protocol
steps = "20" # current step size in current steps protocol
analysis_type = "full" # full or stats
finished = False # whether analysis is finished
italics = '' # whether any group should be italicized in plots



def analyze_data():

    # Make italicized plot labels if wanted
    group2 = '{}'.format(f2)
    group1 = '{}'.format(f1)
    if italics == 'one':
            group1 = '$\it{' + group1 + '}$'
    elif italics == 'two':
            group2 = '$\it{' + group2 + '}$'
    elif italics == 'both':
            group1 = '$\it{' + group1 + '}$'
            group2 = '$\it{' + group2 + '}$'

    # Get AHP amp and time
    bc_input_wt = path + "\\{}\\Brief_current".format(f1)
    bc_output_wt = path + "\\Results\\{}_ahp.csv".format(f1)

    bc_input_ex = path + "\\{}\\Brief_current".format(f2)
    bc_output_ex = path + "\\Results\\{}_ahp.csv".format(f2)

    # Get input resistances
    vc_input_wt = path + "\\{}\\Membrane_test_vc".format(f1)
    vc_output_wt = path + "\\Results\\{}_input_resistances.csv".format(f1)

    vc_input_ex = path + "\\{}\\Membrane_test_vc".format(f2)
    vc_output_ex = path + "\\Results\\{}_input_resistances.csv".format(f2)

    # Get resting potential
    gf_input_wt = path + "\\{}\\Gap_free".format(f1)
    gf_output_wt = path + "\\Results\\{}_resting_potential.csv".format(f1)

    gf_input_ex = path + "\\{}\\Gap_free".format(f2)
    gf_output_ex = path + "\\Results\\{}_resting_potential.csv".format(f2)

    # Get current steps vs frequency and characteristics of spike train and characteristics of AP
    cc_input_wt = path + "\\{}\\Current_steps".format(f1)
    cc_output_wt1 = path + "\\Results\\{}_current_vs_frequency.csv".format(f1)
    cc_output_wt2 = path + "\\Results\\{}_current_steps.csv".format(f1)
    cc_output_wt3 = path + "\\Results\\{}_sag.csv".format(f1)

    cc_input_ex = path + "\\{}\\Current_steps".format(f2)
    cc_output_ex1 = path + "\\Results\\{}_current_vs_frequency.csv".format(f2)
    cc_output_ex2 = path + "\\Results\\{}_current_steps.csv".format(f2)
    cc_output_ex3 = path + "\\Results\\{}_sag.csv".format(f2)

    if analysis_type == 'full':
        joannas_brief_current.analyze_bc(bc_input_wt, bc_output_wt)
        joannas_brief_current.analyze_bc(bc_input_ex, bc_output_ex)
        joannas_vc_test.get_input_resistance_from_vc(vc_input_wt, vc_output_wt)
        joannas_vc_test.get_input_resistance_from_vc(vc_input_ex, vc_output_ex)
        joannas_IC_gap_free.get_resting_potential_from_gf(gf_input_wt, gf_output_wt)
        joannas_IC_gap_free.get_resting_potential_from_gf(gf_input_ex, gf_output_ex)
        joannas_current_steps.analyze_cc(cc_input_wt, cc_output_wt1, cc_output_wt2, cc_output_wt3)
        joannas_current_steps.analyze_cc(cc_input_ex, cc_output_ex1, cc_output_ex2, cc_output_ex3)

    # Read in data we just analyzed
    curr_steps_ex_df = pd.read_csv(cc_output_ex2)
    curr_steps_wt_df = pd.read_csv(cc_output_wt2)
    # Remove time constants that couldn't be calculated
    curr_steps_ex_df["Time Constant (ms)"] = curr_steps_ex_df["Time Constant (ms)"].replace(0, np.nan)
    curr_steps_wt_df["Time Constant (ms)"] = curr_steps_wt_df["Time Constant (ms)"].replace(0, np.nan)
    # Remove frequency adaptation that couldn't be calculated (less than 10 APs)
    curr_steps_ex_df["SFA10"] = curr_steps_ex_df["SFA10"].replace("None", np.nan)
    curr_steps_wt_df["SFA10"] = curr_steps_wt_df["SFA10"].replace("None", np.nan)
    curr_steps_ex_df["SFAn"] = curr_steps_ex_df["SFAn"].replace("None", np.nan)
    curr_steps_wt_df["SFAn"] = curr_steps_wt_df["SFAn"].replace("None", np.nan)
    curr_steps_ex_df[["SFA10", "SFAn"]] = curr_steps_ex_df[["SFA10", "SFAn"]].apply(pd.to_numeric)
    curr_steps_wt_df[["SFA10", "SFAn"]] = curr_steps_wt_df[["SFA10", "SFAn"]].apply(pd.to_numeric)

    ahp_ex_df = pd.read_csv(bc_output_ex)
    ahp_wt_df = pd.read_csv(bc_output_wt)
    # Remove data points with no hyperpolarization
    ahp_ex_df = ahp_ex_df.loc[~((ahp_ex_df['AHP Amplitude (mV)'] == 0) | (ahp_ex_df['AHP Time (ms)'] == 0) | (ahp_ex_df['AHP Amplitude (mV)'] == 'nan'))]
    ahp_wt_df = ahp_wt_df.loc[~((ahp_wt_df['AHP Amplitude (mV)'] == 0) | (ahp_wt_df['AHP Time (ms)'] == 0) | (ahp_wt_df['AHP Amplitude (mV)'] == 'nan'))]

    res_ex_df = pd.read_csv(vc_output_ex)
    res_wt_df = pd.read_csv(vc_output_wt)

    restPot_ex_df = pd.read_csv(gf_output_ex)
    restPot_wt_df = pd.read_csv(gf_output_wt)

    combined_ex = pd.concat([curr_steps_ex_df, ahp_ex_df, res_ex_df, restPot_ex_df], ignore_index=True)
    combined_wt = pd.concat([curr_steps_wt_df, ahp_wt_df, res_wt_df, restPot_wt_df], ignore_index=True)

    # Add label to data (experimental vs wild type)
    combined_ex['Group'] = '{}'.format(f2)
    combined_wt['Group'] = '{}'.format(f1)
    full_df = pd.concat([combined_ex, combined_wt], ignore_index=True)

    col_names = list(combined_ex.columns.values)

    # Do t-test for each column and write in file
    feature_stat_data = []
    stat_cols = ["Measurement", "{}_mean".format(f2), "{}_stderr".format(f2), "{}_n".format(f2),
            "{}_mean".format(f1), "{}_stderr".format(f1), "{}_n".format(f1), "p-value"]
    for col in col_names:
            if col != "filename" and col != "Group":
                ex = combined_ex[col].dropna()
                wt = combined_wt[col].dropna()
                ex_se = 0
                wt_se = 0
                p_val = 0
                if len(ex) != 0 and len(wt) != 0:
                    ex_se = ex.std() /  math.sqrt(len(ex))
                    wt_se = wt.std() /  math.sqrt(len(wt))
                    t_stat, p_val = ttest_ind(ex, wt, nan_policy="omit")
                    feature_stat_data.append([col, ex.mean(), ex_se, len(ex), wt.mean(), wt_se, len(wt), p_val])

                feature_stat_df = pd.DataFrame(feature_stat_data)
                feature_stat_df.columns = stat_cols
                bh_test, corrected_p, _, _ = multi.multipletests(feature_stat_df["p-value"], method="fdr_bh") 
                feature_stat_df["corrected_p"] = corrected_p
                feature_stat_df.to_csv(path + "\\Results\\statistics.csv", index=False)

                # Make scatter plot for each property
                plt.close('all')
                plt.figure(figsize=(10, 16))
                fig, ax = plt.subplots()
                ax.set_ylabel("{}".format(col))
                ax.errorbar('', ex.mean(), yerr=ex_se, fmt="", color="white") #plots nothing to reduce space between two groups on the plot
                sns.stripplot(data=full_df, x='Group', y='{}'.format(col), hue='Group', palette={'{}'.format(f1) : 'blue', '{}'.format(f2) : 'red'}, legend=None, ax=ax, jitter=0.2)
                ax.errorbar('{}'.format(f2), ex.mean(), yerr=ex_se, fmt="_", color="black", capsize=7, markersize=25, markeredgewidth=2, zorder=5)
                ax.errorbar('{}'.format(f1), wt.mean(), yerr=wt_se, fmt="_", color="black", capsize=7, markersize=25, markeredgewidth=2, zorder=5)
                ax.set_box_aspect(2/1)
                ax.errorbar(' ', wt.mean(), yerr=wt_se, fmt="", color="white") #plots nothing to reduce space between two groups on the plot
                if col != 'AP Threshold (mV)' and col != 'Vm (mV)':
                    ax.set_ylim(bottom=0)
                else:
                     ax.set_ylim(top=0)
                ax.spines[['right', 'top']].set_visible(False)
                ax.set(xlabel=None)
                plt.tick_params(axis='x', which='both', bottom=False, labelbottom=True)                     
                plt.xticks(ticks=[0,1,2,3], labels=['', group2, group1, '  '])

                plt.savefig(path + "\\Results\\{}_plot".format(col))
                plt.close('all')

    # Make frequency vs current data for plotting and stats
    def make_cvf_df(file):
        c_vs_f_data = pd.read_csv(file, index_col=False)
        vals_df = c_vs_f_data.filter(regex='Values')
        val_cols = vals_df.columns
        vals_df['n'] = vals_df.count(axis=1)

        # Get current
        max_steps = vals_df.count().max()
        min_curr = float(min_c)
        curr_step = float(steps)
        max_curr = min_curr + (curr_step * max_steps)
        currs = np.arange(min_curr, max_curr, curr_step)
        vals_df['Current'] = currs

        # Calculate mean and stderr
        vals_df['mean'] = vals_df[val_cols].mean(axis=1)
        vals_df['se'] = vals_df[val_cols].sem(axis=1)
        vals_df['ci'] = 1.96 * (vals_df['se']/np.sqrt(vals_df['n']))

        return vals_df

    # Make fold-rheobase data for plotting and stats
    def make_rheo_df(file, name):
            c_vs_f_data = pd.read_csv(file, index_col=False)
            val_cols = c_vs_f_data.filter(regex='Values').columns
            curr_cols = c_vs_f_data.filter(regex='.abf').columns

            # Get current
            max_steps = c_vs_f_data.count().max()
            curr_step = float(steps)
            min_curr = float(min_c)
            max_curr = min_curr + (curr_step * max_steps)
            currs = np.arange(min_curr, max_curr, curr_step)

            # Convert currents to fold rheobase
            i = 0
            rheobase = c_vs_f_data[val_cols].ne(0).idxmax() # get index of first non-zero frequency for each cell
            for col in curr_cols:
                c_vs_f_data.loc[:, col] = currs
                c_vs_f_data.loc[:, col] /= ((rheobase.iloc[i] * curr_step) + min_curr)
                c_vs_f_data.loc[:, col] = c_vs_f_data.loc[:, col].shift(periods=-rheobase.iloc[i])
                c_vs_f_data.loc[:, val_cols[i]] = c_vs_f_data.loc[:, val_cols[i]].shift(periods=-rheobase.iloc[i])
                i += 1

            # Save fold-rheobase data
            c_vs_f_data.to_csv(path + "\\Results\\{}_rheo_vs_frequency.csv".format(name), index=False)
            
            # Make fold-rheobase data for plotting (only integer folds, up to 10)
            points = []
            for i in range(len(curr_cols)):
                for j in range(c_vs_f_data.loc[:, curr_cols[i]].shape[0]):
                    curr = c_vs_f_data.loc[:, curr_cols[i]][j]
                    points.append((curr, c_vs_f_data.loc[:, val_cols[i]][j]))

            point_dict = defaultdict(list)
            folds = []
            for i, j in points: 
                if i in [1,2,3,4,5,6,7,8,9,10]:
                    point_dict[i].append(j)
                    folds.append(i)

            folds = list(set(folds))
            folds.sort()
            final_df = pd.DataFrame.from_dict(point_dict, orient='index')
            new_col_names = []
            for i in range(len(final_df.columns)):
                new_col_names.append("Values_{}".format(i))
            final_df.columns = new_col_names

            # Calculate mean and stderr
            final_df['n'] = final_df.count(axis=1)
            final_df['mean'] = final_df.mean(axis=1)
            final_df['se'] = final_df.sem(axis=1)
            final_df['ci'] = 1.96 * (final_df['se']/np.sqrt(final_df['n']))

            final_df.insert(0, 'Fold Rheobase', folds)
            
            return final_df

    freq_df_wt = make_cvf_df(cc_output_wt1)
    freq_df_ex = make_cvf_df(cc_output_ex1)

    rheo_df_wt = make_rheo_df(cc_output_wt1, f1)
    rheo_df_ex = make_rheo_df(cc_output_ex1, f2)

    # Plot and statistics of x vs frequency
    def plot_and_stats(df_ex, df_wt, x_name):

        # FDR corrected stats
        num_tests = df_ex.shape[0]
        ex_vals = df_ex.filter(regex="Values")
        wt_vals = df_wt.filter(regex="Values")
        stat_data = []
        col_names = [x_name, "{}_mean".format(f2), "{}_stderr".format(f2), "{}_n".format(f2),
                          "{}_mean".format(f1), "{}_stderr".format(f1), "{}_n".format(f1), "p-value"]
        for i in range(num_tests):
            t_stat, p_val = ttest_ind(np.array(ex_vals.iloc[i]), np.array(wt_vals.iloc[i]), nan_policy="omit") #set equal_var to false to use Welch's test (unequal variances)
            if not np.isnan(p_val):
                stat_data.append([df_ex[x_name].iloc[i], df_ex['mean'].iloc[i], df_ex['se'].iloc[i], df_ex['n'].iloc[i], 
                            df_wt['mean'].iloc[i], df_wt['se'].iloc[i], df_wt['n'].iloc[i], p_val])

        stats_df = pd.DataFrame(stat_data)
        stats_df.columns = col_names
        bh_test, corrected_p, _, _ = multi.multipletests(stats_df["p-value"], method="fdr_bh")
        stats_df["corrected_p"] = corrected_p
        stats_df.to_csv(path + "\\Results\\{}_freq_stats.csv".format(x_name), index=False)


        # Make freq plot
        fig, ax = plt.subplots()
        ax.plot(df_wt[x_name], df_wt['mean'], color="darkcyan", label=group1)
        x_label = x_name
        if x_name == 'Current':
            x_label += ' (pA)'
        ax.set_xlabel(x_label)
        ax.set_ylabel("Frequency (Hz)")
        ax.fill_between(df_wt[x_name], (df_wt['mean']-df_wt['ci']), (df_wt['mean']+df_wt['ci']), facecolor='b', alpha=0.3)
        ax.plot(df_ex[x_name], df_ex['mean'], color='firebrick', label=group2)
        ax.fill_between(df_ex[x_name], (df_ex['mean']-df_ex['ci']), (df_ex['mean']+df_ex['ci']), facecolor='r', alpha=0.3)
        ax.legend()
        ax.set_title("{} vs Frequency of Action Potentials".format(x_name))
        plt.savefig(path + "\\Results\\{}_vs_frequency_plot".format(x_name))
        plt.close('all')

    plot_and_stats(freq_df_ex, freq_df_wt, "Current")
    plot_and_stats(rheo_df_ex, rheo_df_wt, "Fold Rheobase")

    global finished
    finished = True



def get_data():
        global path
        path = entry_path.get()
        global f1
        f1 = entry_f1.get()
        global f2
        f2 = entry_f2.get()
        global min_c
        min_c = entry_min.get()
        global steps
        steps = entry_steps.get()
        global analysis_type
        analysis_type = entry_type.get()
        global italics
        italics = it_entry.get()
        window.title("Analyzing")
        for widgets in window.winfo_children():
            widgets.destroy()
        label = tk.Label(window, text="Analyzing, please wait")
        label.pack()
        analysis = threading.Thread(target=analyze_data)
        analysis.daemon = True
        analysis.start()
        while finished == False and analysis.is_alive():
            window.update()
        if finished:
            window.title("Finished")
            label['text'] = "Analysis Completed"
        else:
            window.title("Crashed")
            label['text'] = "Analysis was not completed. Something went wrong. Check the instructions and try again."
        button = tk.Button(window, text="Done", width=20, command=done, activebackground='#78d6ff')
        button.pack()


window = tk.Tk()
window.geometry("800x400")
window.title("Input Data")

def done():
    window.quit()
    window.destroy()
window.protocol("WM_DELETE_WINDOW", done)

path_label = tk.Label(window, text="Enter Path to Folder Containing Data")
entry_path = tk.Entry(window, width=40)
group1_label = tk.Label(window, text="Group 1 (Blue) - Must Match Folder Name")
entry_f1 = tk.Entry(window, width=40)
group2_label = tk.Label(window, text="Group 2 (Red) - Must Match Folder Name")
entry_f2 = tk.Entry(window, width=40)
min_label = tk.Label(window, text="Minimum Current in Current Steps Protocol")
entry_min = tk.Entry(window, width=40)
steps_label = tk.Label(window, text="Step Size in Current Steps Protocol")
entry_steps = tk.Entry(window, width=40)
type_label = tk.Label(window, text="Perfom Whole Analysis->full or Only Statistics->stats")
entry_type = tk.Entry(window, width=40)
it_label = tk.Label(window, text="Italicize Group 1->one or Group 2->two or Both->both or None->no")
it_entry = tk.Entry(window, width=40)

path_label.grid(row=0, column=1)
entry_path.grid(row=0, column=2)
group1_label.grid(row=1, column=1)
entry_f1.grid(row=1, column=2)
group2_label.grid(row=2, column=1)
entry_f2.grid(row=2, column=2)
min_label.grid(row=3, column=1)
entry_min.grid(row=3, column=2)
steps_label.grid(row=4, column=1)
entry_steps.grid(row=4, column=2)
type_label.grid(row=5, column=1)
entry_type.grid(row=5, column=2)
it_label.grid(row=6, column=1)
it_entry.grid(row=6, column=2)

btn = tk.Button(window, text="Enter", width=20, command=get_data, activebackground='#78d6ff')
btn.grid(row=7, column=2)

window.mainloop()
