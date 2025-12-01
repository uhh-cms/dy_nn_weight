import numpy as np
import matplotlib.pyplot as plt
import math
import awkward as ak
import pandas as pd
import os
import correctionlib


#creating a path to access exemplary data files
mypath = "/data/dust/user/riegerma/hh2bbtautau/dy_dnn_data/inputs/22pre_v14/"
example_file = "w_lnu_1j_pt100to200_amcatnlo.parquet"
fullpath = f"{mypath}{example_file}"
example_array = ak.from_parquet(fullpath)
#make arrays with the extracted data
ll_pt = ak.from_parquet(fullpath)["ll_pt"]
ll_eta = ak.from_parquet(fullpath)["ll_eta"]
event_weight = ak.from_parquet(fullpath)["event_weight"]

#change the event weights randomly for testing purposes
new_event_weight = np.ones(115, dtype=float)
for i in range(len(event_weight)):
    if ll_pt[i]<50:
        new_event_weight[i]=abs(event_weight[i])*0.8
    else:
        new_event_weight[i]=abs(event_weight[i])*1.2










variables = example_array.fields
print(variables)

def create_full_arrays(variable):
    filelist = os.listdir(mypath)

    data_list = []
    dy_list = []
    mc_list = []

    for file in filelist:
        #extract the values from the file if they exist
        try:
            values = ak.from_parquet(f"{mypath}{file}")[variable]
        #for the data that don't have event weights create event weights with the value 1
        except:
            ll_len = len(ak.from_parquet(f"{mypath}{file}")["ll_pt"])
            values = ak.Array(np.ones(ll_len, dtype=float))
        #append values to the correct list
        if file.startswith("data"):
            data_list.append(values)
        elif file.startswith("dy"):
            dy_list.append(values)
        else:
            mc_list.append(values)


    # helper to safely concatenate awkward arrays and return a numpy array
    def _concat_to_numpy(lst):
        if not lst:
            return np.array([], dtype=float)
        concatenated = ak.concatenate(lst)
        try:
            return ak.to_numpy(concatenated)
        except Exception:
            return np.asarray(ak.to_list(concatenated), dtype=float)

    data_arr = _concat_to_numpy(data_list)
    dy_arr = _concat_to_numpy(dy_list)
    mc_arr = _concat_to_numpy(mc_list)

    print(f"{variable} Arrays successfully created.")

    return data_arr, dy_arr, mc_arr







#define a function for plotting histograms

def plot_histogram(variable, file_version, dy_weights, n_bins=None):
    data_arr, dy_arr, mc_arr = create_full_arrays(variable)
    _, _, mc_weights = create_full_arrays("event_weight")

    # If no bins are provided create default bins
    if n_bins is None:
        if variable == "ll_pt" or variable == "bb_pt" or variable == "llbb_pt" or variable == "lep1_pt" or variable == "jet1_pt":
            n_bins = np.linspace(0, 400, 41)
        elif variable == "ll_eta" or variable == "bb_eta" or variable == "llbb_eta" or variable == "lep1_eta" or variable == "jet1_eta":
            n_bins = np.linspace(-math.pi, math.pi, 21)
        elif variable == "ll_phi" or variable == "bb_phi" or variable == "llbb_phi" or variable == "lep1_phi" or variable == "jet1_phi":
            n_bins = np.linspace(-math.pi, math.pi, 21)
        elif variable == "ll_mass" or variable == "bb_mass":
            n_bins = np.linspace(0, 400, 41)
        elif variable == "llbb_mass":
            n_bins = np.linspace(0, 800, 41)
        elif variable == "n_jet" or variable == "n_btag_pnet" or variable == "n_btag_pnet_hhb":
            n_bins = np.linspace(0, 10, 11)
        else:
            n_bins = np.linspace(min(min(data_arr), min(min(dy_arr), min(mc_arr))), max(max(data_arr), max(max(dy_arr), max(mc_arr))), 21)


    fig, ax = plt.subplots()

    # Plotting MC and DY histograms
    # using clip to add overflow events to the last bin
    ax.hist([np.clip(mc_arr, n_bins[0], n_bins[-1]), np.clip(dy_arr, n_bins[0], n_bins[-1])],
            bins=n_bins,
            histtype="barstacked",
            color=['red', 'blue'],
            label=['MC', 'DY'],
            weights=[mc_weights, dy_weights])

    # Data: per-bin counts, plot as black dots at bin centers
    data_counts, bins = np.histogram(np.clip(data_arr, n_bins[0], n_bins[-1]), bins=n_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(bin_centers, data_counts, 'ko', label='Data')


    ax.set_xlabel(variable)
    ax.set_ylabel('Events / bin')
    ax.legend()
    ax.set_title(f'{variable} histogram')
    plt.savefig(f"{variable}_{file_version}.pdf")
    plt.show()






# Plot original histogram with old DY event weights
# extract full arrays for ll_pt and event weights
data_ll_pt, dy_ll_pt, mc_ll_pt = create_full_arrays("ll_pt")
data_event_weight, dy_event_weight, mc_event_weight = create_full_arrays("event_weight")

# define binning
n_bins = np.linspace(0, 400, 21)

#plot_histogram("ll_pt", "old_event_weights", dy_event_weight, n_bins)








# # Plot histogram with new DY event weights
# # Calculate new event weights for DY
# data_counts, _ = np.histogram(np.clip(data_ll_pt, n_bins[0], n_bins[-1]), bins=n_bins)
# dy_counts, _ = np.histogram(np.clip(dy_ll_pt, n_bins[0], n_bins[-1]), bins=n_bins, weights=dy_event_weight)
# mc_counts, _ = np.histogram(np.clip(mc_ll_pt, n_bins[0], n_bins[-1]), bins=n_bins, weights=mc_event_weight)
# # If the DY counts are zero, avoid division by zero
# with np.errstate(divide='ignore', invalid='ignore'):
#     bin_factor = (data_counts - mc_counts)/ dy_counts
# bad = (~np.isfinite(bin_factor)) | (dy_counts == 0)
# bin_factor[bad] = 1.0


# # Map each dy_ll_pt to its corresponding bin factor
# dy_bin_idx = np.digitize(dy_ll_pt, bins=n_bins) - 1
# dy_bin_idx = np.clip(dy_bin_idx, 0, len(bin_factor)-1)

# new_dy_event_weight = dy_event_weight * bin_factor[dy_bin_idx]

# # Plot the bin factors
# plt.plot(bin_factor, 'bo')
# plt.xlabel("ll_pt")
# plt.ylabel("Ratio")
# plt.title("Ratio of new to old DY event weights")
# plt.savefig("dy_event_weight_ratio.pdf")
# plt.show()


#plot_histogram("ll_pt", "new_event_weights", new_dy_event_weight, n_bins)





# Use another method to get DY weights from correctionlib
dy_file = "/afs/desy.de/user/a/alvesand/public/dy_corrections.json.gz"
dy_correction = correctionlib.CorrectionSet.from_file(dy_file)
correction_set = dy_correction["dy_weight"]
# inputs = ['era', 'njets', 'ntags', 'ptll', 'syst']
era = "2022preEE"
syst =  "nom"
# weight = correction_set.evaluate(era,  "n_jet", "n_tag", "ll_pt", syst)




_, dy_n_jet, _ = create_full_arrays("n_jet")
_, dy_n_tag, _ = create_full_arrays("n_btag_pnet")


weight = correction_set.evaluate(era, dy_n_jet, dy_n_tag, dy_ll_pt, syst)
correctionlib_weight = weight * dy_event_weight
# print(correctionlib_weight)

# plot_histogram("bb_mass", "correctionlib_weights", correctionlib_weight)
# plot_histogram("ll_mass", "correctionlib_weights", correctionlib_weight)
# plot_histogram("llbb_mass", "correctionlib_weights", correctionlib_weight)

# Filter the events for different categories
#filtering by channel_id: ee = 4, mumu = 5
def filter_events_by_channel(array, desired_channel_id):
    _, dy_channel_id, _ = create_full_arrays("channel_id")
    print(dy_channel_id)
    filtered_array = np.array([])
    for i in range(len(dy_channel_id)):
        if dy_channel_id[i] == desired_channel_id:
            filtered_array = np.append(filtered_array, array[i])
    print(f"Filtered for channel ID {desired_channel_id}: {len(filtered_array)} entries")
    return filtered_array

filter_events_by_channel(dy_ll_pt, 4)  # for ee channel
filter_events_by_channel(dy_ll_pt, 5)  # for mumu channel

# Filtering by ll_mass 70-110 GeV
def filter_events_by_ll_mass(array, mass_min, mass_max):
    _, dy_ll_mass, _ = create_full_arrays("ll_mass")
    print(dy_ll_mass)
    filtered_array = np.array([])
    for i in range(len(dy_ll_mass)):
        if mass_min <= dy_ll_mass[i] <= mass_max:
            filtered_array = np.append(filtered_array, array[i])
    print(f"Filtered for ll_mass between {mass_min} and {mass_max} GeV: {len(filtered_array)} entries")
    return filtered_array

filter_events_by_ll_mass(dy_ll_pt, 70, 110)  # for ll_mass between 70 and 110 GeV

# Filtering by met < 45 GeV
