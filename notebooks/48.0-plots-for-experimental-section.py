import pickle
import definitions
import matplotlib.pyplot as plt
import src.features.compare_datasets as cd
"""This script is used to generate plots for the experimental section of the report
it includes 1) Frequency domain and 2) Time domain methods"""

to_apply = [
            "order_track"
            ]

channel = "Acc_Carrier"


def save_figure_in_report_folder(name):
    path = definitions.root + "\\reports\\masters_report\\3_experimental_work\\Images\\"
    # plt.savefig(path + name + "20201122.pdf")
    return

def make_tsa_plot(filename, name,offset_frac):
    directory = definitions.root + "\\data\\processed\\" + filename
    with open(directory, 'rb') as p:
        data = pickle.load(p)
        signal = data.dataset[channel].values
        data.compute_tsa(offset_frac, 2/62, signal, plot=True)
        print("RPM",data.info["rpm_sun_ave"]," RPM ",name)
    save_figure_in_report_folder(name)
    return

healthy_full_thickness = "g1_p0_v8_8.pgdata"
tooth_missing_full_thickness = "cycle_2_end.pgdata"
# full_crack_half_thickness = "g1_p7_8.8.pgdata"

#full_crack_half_thickness = "g2_p5_9.2.pgdata" # used for tsa plot
full_crack_half_thickness = "g2_p5_9.0.pgdata" # 1031 used for freq domain
healthy_half_thickness = "g2_p0_9.4.pgdata" # 1056 rpm

#TSA plots
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Full thickness gear
# offset_frac = 0.5/62 # This is an earlier dataset with a different offset fraction required
# make_tsa_plot(healthy_full_thickness, "healthy_full_thickness_tsa", offset_frac)
# make_tsa_plot(tooth_missing_full_thickness, "tooth_missing_full_thickness_tsa", offset_frac=0)
#
# # Half thickness full crack
# offset_frac = 0
# make_tsa_plot(full_crack_half_thickness, "tsa_full_crack", offset_frac)

# FREQUENCY DOMAIN METHODS TOOTH MISSING
# Harmonics used 4,2 and perhaps 1
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# for i in range(1, 7):
for i in [4,2,1]:
    cd.squared_spectrum_at_harmonics_multi_plot(tooth_missing_full_thickness, healthy_full_thickness, i,
                                            "missing_tooth" +
                                             str(i))

    cd.ses_multi_plot(tooth_missing_full_thickness,healthy_full_thickness, i, "missing_tooth" +
                                                str(i))

# for i in range(1, 7):
#     cd.squared_spectrum_at_harmonics_multi_plot(full_crack_half_thickness, healthy_half_thickness, i,
#                                                 "cracked" + str(i))
#
#     cd.ses_multi_plot(full_crack_half_thickness, healthy_half_thickness, i, "cracked" +
#                       str(i))
