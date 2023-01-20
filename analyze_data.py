import tkinter as tk
from joannas_vc_test import get_input_resistance_from_vc

window = tk.Tk()
window.title("Input Data")

wt_path = ""
ex_path = ""

label_wt = tk.Label(window, text="Enter Wild Type Data Path")
label_wt.pack()

entry_wt = tk.Entry(window, width=40)
entry_wt.pack()

label_ex = tk.Label(window, text="Enter Experimental Data Path")
label_ex.pack()

entry_ex = tk.Entry(window, width=40)
entry_ex.pack()


def submit():
    global wt_path
    wt_path = entry_wt.get()
    global ex_path
    ex_path = entry_ex.get()
    window.destroy()


tk.Button(window, text="Enter", width=20, command=submit).pack(pady=20)

window.mainloop()

window = tk.Tk()
window.title("Analyzing Data")

label = tk.Label(window, text="Analyzing")
label.pack()

vc_input_wt = wt_path + "\\VC"
vc_output_wt = wt_path + "\\wt_input_resistances.csv"
get_input_resistance_from_vc(vc_input_wt, vc_output_wt)

vc_input_ex = ex_path + "\\VC"
vc_output_ex = ex_path + "\\ex_input_resistances.csv"
get_input_resistance_from_vc(vc_input_ex, vc_output_ex)

label.destroy()
label = tk.Label(window, text="Click Done")
label.pack()


def done():
    window.destroy()


tk.Button(window, text="Done", width=20, command=done).pack(pady=20)

window.mainloop()
