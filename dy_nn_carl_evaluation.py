from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import awkward as ak
import order as od
import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from modules.hbt.hbt.config.analysis_hbt import analysis_hbt
from modules.hbt.modules.columnflow.columnflow.plotting.plot_functions_1d import plot_variable_stack
from modules.hbt.modules.columnflow.columnflow.hist_util import create_hist_from_variables, fill_hist  # noqa: E501

# --------------------------------------------------------------------------------------------------

# SETUP of files
year = "22pre_v14"
config_inst = analysis_hbt.get_config(year)
mypath = f"/data/dust/user/riegerma/hh2bbtautau/dy_dnn_data/inputs_prod20_vbf/{year}/"  # noqa: E501
filelist = os.listdir(mypath)
example_file = "w_lnu_1j_pt100to200_amcatnlo.parquet"
fullpath = f"{mypath}{example_file}"
example_array = ak.from_parquet(fullpath)
variables = example_array.fields
print(variables)

syst = "nom"
era = year.replace("22", "2022").replace("_v14", "EE")
channel_id = 5  # mumu channel
# --------------------------------------------------------------------------------------------------

# PREPARE VARIABLES
ll_mass = config_inst.variables.n.dilep_mass
ll_pt = config_inst.variables.n.dilep_pt
ll_eta = config_inst.variables.n.dilep_eta
ll_phi = config_inst.variables.n.dilep_phi

bb_mass = config_inst.variables.n.dibjet_mass
bb_pt = config_inst.variables.n.dibjet_pt
bb_eta = config_inst.variables.n.dibjet_eta
bb_phi = config_inst.variables.n.dibjet_phi

n_jet = config_inst.variables.n.njets
n_btag_pnet = config_inst.variables.n.nbjets_pnet

met_pt = config_inst.variables.n.met_pt
met_phi = config_inst.variables.n.met_phi

jet1_pt = config_inst.variables.n.jet1_pt
jet1_eta = config_inst.variables.n.jet1_eta
jet1_phi = config_inst.variables.n.jet1_phi

if channel_id == 4:
    lep1_pt = config_inst.variables.n.e1_pt
    lep1_eta = config_inst.variables.n.e1_eta
    lep1_phi = config_inst.variables.n.e1_phi
elif channel_id == 5:
    lep1_pt = config_inst.variables.n.mu1_pt
    lep1_eta = config_inst.variables.n.mu1_eta
    lep1_phi = config_inst.variables.n.mu1_phi
else:
    raise ValueError("Invalid channel_id. Must be 4 (ee) or 5 (mumu).")

variables_to_plot = [
    ll_mass, ll_pt, ll_eta, ll_phi,
    bb_mass, bb_pt, bb_eta, bb_phi,
    n_jet, n_btag_pnet,
    met_pt, met_phi,
    jet1_pt, jet1_eta, jet1_phi,
    lep1_pt, lep1_eta, lep1_phi]

# get processes
bkg_process = od.Process(name="mc", id="+", color1="#e76300", label="MC")
dy_process = config_inst.get_process("dy")
data_process = config_inst.get_process("data")

# create a full dictionary of the variables
temp_storage = {
    "data": {var: [] for var in variables},
    "dy": {var: [] for var in variables},
    "mc": {var: [] for var in variables},
}

for file in filelist:
    full_ak_array = ak.from_parquet(f"{mypath}{file}")
    for var in variables:
        # extract the variable values from the file
        if var != "event_weight" or not file.startswith("data"):
            values = full_ak_array[var]
        else:
            # create event weights = 1 for data
            ll_len = len(full_ak_array["ll_pt"])
            values = ak.Array(np.ones(ll_len, dtype=float))

        # append values to the correct list
        if file.startswith("data"):
            temp_storage["data"][var].append(values)
        elif file.startswith("dy"):
            temp_storage["dy"][var].append(values)
        else:
            temp_storage["mc"][var].append(values)

# --------------------------------------------------------------------------------------------------

# HELPER FUNCTIONS
def _concat_to_numpy(lst):
    if not lst:
        return np.array([], dtype=float)
    concatenated = ak.concatenate(lst)
    try:
        return ak.to_numpy(concatenated)
    except Exception:
        return np.asarray(ak.to_list(concatenated), dtype=float)

def filter_events_by_channel(data_arr, dy_arr, mc_arr, channel_id, return_masks=False, b_tag_cut=None, n_jet_cut=None):
    """
    Function to filter the events by channel id and DY region cuts
    -> desired_channel_id: ee = 4, mumu = 5
    """
    data_channel_id, dy_channel_id, mc_channel_id = variable_list["channel_id"]  # noqa: E501
    data_ll_mass, dy_ll_mass, mc_ll_mass = variable_list["ll_mass"]  # noqa: E501
    data_met, dy_met, mc_met = variable_list["met_pt"]
    # view plots with cuts in discrete variables
    if b_tag_cut == 2:
        data_n_btag, dy_n_btag, mc_n_btag = variable_list["n_btag_pnet"]
        data_n_btag_mask = data_n_btag >= b_tag_cut
        dy_n_btag_mask = dy_n_btag >= b_tag_cut
        mc_n_btag_mask = mc_n_btag >= b_tag_cut
    elif b_tag_cut in [0, 1]:
        data_n_btag, dy_n_btag, mc_n_btag = variable_list["n_btag_pnet"]
        data_n_btag_mask = data_n_btag == b_tag_cut
        dy_n_btag_mask = dy_n_btag == b_tag_cut
        mc_n_btag_mask = mc_n_btag == b_tag_cut
    else:
        data_n_btag_mask = np.ones_like(data_channel_id, dtype=bool)
        dy_n_btag_mask = np.ones_like(dy_channel_id, dtype=bool)
        mc_n_btag_mask = np.ones_like(mc_channel_id, dtype=bool)
    
    if n_jet_cut == 4:
        data_n_jet, dy_n_jet, mc_n_jet = variable_list["n_jet"]
        data_n_jet_mask = data_n_jet >= n_jet_cut
        dy_n_jet_mask = dy_n_jet >= n_jet_cut
        mc_n_jet_mask = mc_n_jet >= n_jet_cut
    elif n_jet_cut in [2, 3]:
        data_n_jet, dy_n_jet, mc_n_jet = variable_list["n_jet"]
        data_n_jet_mask = data_n_jet == n_jet_cut
        dy_n_jet_mask = dy_n_jet == n_jet_cut
        mc_n_jet_mask = mc_n_jet == n_jet_cut
    else:
        data_n_jet_mask = np.ones_like(data_channel_id, dtype=bool)
        dy_n_jet_mask = np.ones_like(dy_channel_id, dtype=bool)
        mc_n_jet_mask = np.ones_like(mc_channel_id, dtype=bool)

    # create masks
    data_id_mask = data_channel_id == channel_id
    dy_id_mask = dy_channel_id == channel_id
    mc_id_mask = mc_channel_id == channel_id

    data_ll_mask = (data_ll_mass >= 70) & (data_ll_mass <= 110)
    dy_ll_mask = (dy_ll_mass >= 70) & (dy_ll_mass <= 110)
    mc_ll_mask = (mc_ll_mass >= 70) & (mc_ll_mass <= 110)

    data_met_mask = data_met < 45
    dy_met_mask = dy_met < 45
    mc_met_mask = mc_met < 45    

    # combine masks
    data_mask = data_id_mask & data_ll_mask & data_met_mask & data_n_jet_mask & data_n_btag_mask
    dy_mask = dy_id_mask & dy_ll_mask & dy_met_mask & dy_n_jet_mask & dy_n_btag_mask
    mc_mask = mc_id_mask & mc_ll_mask & mc_met_mask & mc_n_jet_mask & mc_n_btag_mask

    if return_masks:
        return data_arr[data_mask], dy_arr[dy_mask], mc_arr[mc_mask], data_mask, dy_mask, mc_mask
    else:
        return data_arr[data_mask], dy_arr[dy_mask], mc_arr[mc_mask]

def hist_function(var_instance, data, weights):
    h = create_hist_from_variables(var_instance, weight=True)
    fill_hist(h, {var_instance.name: data, "weight": weights})
    return h

def plot_function(var_instance: od.Variable, file_version: str, dy_weights=None, filter_events: bool = True, channel_id=None, b_tag_cut=None, n_jet_cut=None):  # noqa: E501
    var_name = var_instance.name.replace("dilep", "ll").replace("dibjet", "bb").replace("njets", "n_jet").replace("nbjets", "n_btag").replace("e1", "lep1").replace("mu1", "lep1")

    # get variable arrays and original event weights from files
    data_arr, dy_arr, mc_arr = variable_list[var_name]
    data_weights, dy_weights_original, mc_weights = variable_list["event_weight"]  # noqa: E501

    # use original DY weights if none are provided
    if dy_weights is None:
        print(f"Saved {var_name} plot --> Using original DY event weights!")
        dy_weights = dy_weights_original
    else:
        print(f"Saved {var_name} plot --> Using updated DY event weights!")
        dy_weights = dy_weights

    # filter arrays considering mumu channel id and DY region cuts
    if filter_events:
        data_arr, dy_arr, mc_arr = filter_events_by_channel(data_arr, dy_arr, mc_arr, channel_id, False, b_tag_cut, n_jet_cut)  # noqa: E501
        data_weights, dy_weights, mc_weights = filter_events_by_channel(  # noqa: E501
            data_weights, dy_weights, mc_weights, channel_id, False, b_tag_cut, n_jet_cut
        )

    # update binning in histograms
    data_hist = hist_function(var_instance, data_arr, data_weights)
    dy_hist = hist_function(var_instance, dy_arr, dy_weights)
    mc_hist = hist_function(var_instance, mc_arr, mc_weights)
    hists_to_plot = {
        dy_process: dy_hist,
        data_process: data_hist,
        bkg_process: mc_hist,
    }

    # create figure
    fig, _ = plot_variable_stack(
        hists=hists_to_plot,
        config_inst=config_inst,
        category_inst=config_inst.get_category("dyc"),
        variable_insts=[var_instance,],
        shift_insts=[config_inst.get_shift("nominal")],
    )

    plt.savefig(f"plot_{file_version}.pdf")

def plot_2d_function(var_instance: od.Variable, file_version: str, dy_weights=None, channel_id=None):  # noqa: E501
    var_name = var_instance.name.replace("dilep", "ll").replace("dibjet", "bb").replace("njets", "n_jet").replace("nbjets", "n_btag").replace("e1", "lep1").replace("mu1", "lep1")  # noqa: E501
    
    if dy_weights is None:
        dy_weights = variable_list["event_weight"][1]
    else:
        dy_weights = dy_weights

    _, dy_arr_filtered, _, _, dy_mask, _ = filter_events_by_channel(*variable_list[var_name], channel_id, return_masks=True)
    dy_weights_filtered = dy_weights[dy_mask]

    y_max = np.percentile(dy_weights_filtered, 99) * 1.2
    y_min = np.percentile(dy_weights_filtered, 1) * 0.8
    x_max = np.percentile(dy_arr_filtered, 99) * 1.2
    x_min = np.percentile(dy_arr_filtered, 1) * 0.8

    plt.figure(figsize=(10,8))
    counts, xedges, yedges, im = plt.hist2d(
        dy_arr_filtered, 
        dy_weights_filtered, 
        bins=(60, 60), 
        range=[[x_min, x_max], [y_min, y_max]],
        norm=mcolors.LogNorm(),
        cmap="viridis"
    )

    color_bar =plt.colorbar(im)
    color_bar.set_label('Number of Events', fontsize=16)
    color_bar.ax.tick_params(labelsize=14)
    plt.xlabel(f"{var_instance.x_title} ({var_instance.unit})" if var_instance.unit else var_instance.x_title, fontsize=16)
    plt.ylabel("DY weights", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig(f"plot_{file_version}_{var_name}_correlation.png", dpi=600)
    print(f"Saved 2D correlation plot for {var_name} with DY weights!")

def plot_all_variables(file_version, dy_weights=None, channel_id=None):
    for var in variables_to_plot:
        var_name = var.name.replace("dilep", "ll").replace("dibjet", "bb").replace("njets", "n_jet").replace("nbjets", "n_btag").replace("e1", "lep1").replace("mu1", "lep1")  # noqa: E501
        plot_function(var, f"{file_version}_{var_name}", dy_weights=dy_weights, channel_id=channel_id)

def plot_calibration(model, data_loader, dy_loader, file_name: str):
    model.eval()
    data_outputs = []
    dy_outputs = []

    with torch.no_grad():
        for batch in data_loader:
            data_outputs.append(model(batch[0]))
        for batch in dy_loader:
            dy_outputs.append(model(batch[0]))
    data_outputs = torch.cat(data_outputs).numpy().flatten()
    dy_outputs = torch.cat(dy_outputs).numpy().flatten()
    
    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    n_total_data = len(data_outputs)
    n_total_dy = len(dy_outputs)

    ratios = []
    for i in range(len(bins)-1):
        n_i_data = np.sum((data_outputs >= bins[i]) & (data_outputs < bins[i+1]))
        n_i_dy = np.sum((dy_outputs >= bins[i]) & (dy_outputs < bins[i+1]))

        p_i_dy = n_i_dy / n_total_dy if n_total_dy > 0 else 0.0
        p_i_data = n_i_data / n_total_data if n_total_data > 0 else 0.0

        ratio = p_i_data / (p_i_data + p_i_dy) if p_i_data + p_i_dy > 0 else 0.0
        ratios.append(ratio)

    plt.figure(figsize=(8,8))
    plt.plot(bin_centers, ratios, marker='o', linestyle='-', label='Model Calibration', color='darkblue')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration', color='gray')
    plt.xlabel('Classifier Output', fontsize=16)
    plt.ylabel('Ratio ($\\frac{P_{Data}}{(P_{DY} + P_{Data})}$)', fontsize=16)
    plt.title('Calibration Plot of the Classifier', fontsize=16)
    plt.legend(fontsize=14)
    plt.savefig(f"classifier_calibration_{file_name}.png", dpi=300)

    print(f"Saved DY classifier calibration plot")


# --------------------------------------------------------------------------------------------------

# PREPARE DATA FOR THE NN
# fill the variable list with arrays for each variable
variable_list = {}
for var in variables:
    data_arr = _concat_to_numpy(temp_storage["data"][var])
    dy_arr = _concat_to_numpy(temp_storage["dy"][var])
    mc_arr = _concat_to_numpy(temp_storage["mc"][var])

    variable_list[var] = [data_arr, dy_arr, mc_arr]


input_variables = ["ll_pt", "n_jet", "n_btag_pnet", "bb_pt", "bb_mass"]
data_variables = []
dy_variables = []
dy_variables_ee = []

for var in input_variables:
    _, dy_var, _ = filter_events_by_channel(*variable_list[var], channel_id)
    dy_variables.append(torch.tensor(dy_var, dtype=torch.float)[:, None])
    _, dy_var_ee, _ = filter_events_by_channel(*variable_list[var], channel_id=4)
    dy_variables_ee.append(torch.tensor(dy_var_ee, dtype=torch.float)[:, None])
    data_var, _, _ = filter_events_by_channel(*variable_list[var], channel_id)
    data_variables.append(torch.tensor(data_var, dtype=torch.float)[:, None])

data_inputs = torch.cat(data_variables, dim=1)
dy_inputs = torch.cat(dy_variables, dim=1)
dy_inputs_ee = torch.cat(dy_variables_ee, dim=1)

# normalize inputs
scaling_params = torch.load("input_scaling_params_v2.pth")
input_means = scaling_params["input_means"]
input_stds = scaling_params["input_stds"]
data_inputs = (data_inputs - input_means) / (input_stds + 1e-8)
dy_inputs = (dy_inputs - input_means) / (input_stds + 1e-8)
dy_inputs_ee = (dy_inputs_ee - input_means) / (input_stds + 1e-8)

# wrap in dataset
data_dataset = TensorDataset(data_inputs)
dy_dataset = TensorDataset(dy_inputs)
dy_dataset_ee = TensorDataset(dy_inputs_ee)


# ----------------------------------------------------------------------------------
# Load the NN
# use a simple feedforward NN with 5-128-64-64-1 architecture, LeakyReLu activation
class DYClassifierNN(nn.Module):
    def __init__(self):
        super(DYClassifierNN, self).__init__()
        self.fc1 = nn.Linear(5, 128)   # input layer to hidden layer
        #self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(128, 64)  # hidden layer to hidden layer
        #self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)  # hidden layer to hidden layer
        #self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)   # hidden layer to output
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        #x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

model = DYClassifierNN()
model.load_state_dict(torch.load("dy_classifier_weights_v2.pth"))
model.eval()

full_loader = DataLoader(dy_dataset, batch_size=16384, shuffle=False)
full_loader_ee = DataLoader(dy_dataset_ee, batch_size=16384, shuffle=False)
full_data_loader = DataLoader(data_dataset, batch_size=16384, shuffle=False)

plot_calibration(model, full_data_loader, full_loader, "full_dataset")
# ----------------------------------------------------------------------------------

# UPDATE DY EVENT WEIGHTS BASED ON NN OUTPUT

dy_corrections_filtered = []
outputs = []
with torch.no_grad():
    for batch in full_loader:
        dy_inputs = batch[0]
        dy_outputs = model(dy_inputs)
        outputs.append(dy_outputs)
        weights = dy_outputs / (1 - dy_outputs + 1e-8)  # avoid division by zero
        dy_corrections_filtered.append(weights)

outputs = torch.cat(outputs).numpy().flatten()

# compare sum of data weights and new dy weights
data_event_weights, dy_event_weights, mc_event_weights, _, dy_mask, _ = filter_events_by_channel(*variable_list["event_weight"], channel_id, return_masks=True)  # noqa: E501
total_data_mc_weight = np.sum(data_event_weights) - np.sum(mc_event_weights)
dy_corrections_filtered = torch.cat(dy_corrections_filtered).numpy().flatten()
new_dy_weights_filtered = dy_corrections_filtered * dy_event_weights
total_new_dy_weight = np.sum(new_dy_weights_filtered)
print(f"Average new DY event weight: {np.mean(new_dy_weights_filtered):.6f}")
print(new_dy_weights_filtered[:100])

scaling_factor_carl = total_data_mc_weight / total_new_dy_weight
new_dy_weights_filtered = new_dy_weights_filtered * scaling_factor_carl
new_dy_weights = variable_list["event_weight"][1].copy()
new_dy_weights[dy_mask] = new_dy_weights_filtered

print("----------- IN THE MUMU CHANNEL -----------")
print(f'Sum of original DY event weights: {np.sum(dy_event_weights):.6f}')
print(f'Sum of new DY event weights: {total_new_dy_weight:.6f}')
print(f'Target sum (data - other mc): {total_data_mc_weight:.6f}')
print(f'Scaling factor applied to new DY weights: {scaling_factor_carl:.6f}')
# ----------------------------------------------------------------------------------

# UPDATE DY EVENT WEIGHTS BASED ON NN OUTPUT - EE CHANNEL

dy_corrections_ee_filtered = []
with torch.no_grad():
    for batch in full_loader_ee:
        dy_inputs_ee = batch[0]
        dy_outputs_ee = model(dy_inputs_ee)
        weights_ee = dy_outputs_ee / (1 - dy_outputs_ee + 1e-8)
        dy_corrections_ee_filtered.append(weights_ee)

data_event_weights_ee, dy_event_weights_ee, mc_event_weights_ee, _, dy_mask_ee, _ = filter_events_by_channel(*variable_list["event_weight"], channel_id=4, return_masks=True)  # noqa: E501
total_data_mc_weight_ee = np.sum(data_event_weights_ee) - np.sum(mc_event_weights_ee)
dy_corrections_ee_filtered = torch.cat(dy_corrections_ee_filtered).numpy().flatten()
new_dy_weights_ee_filtered = dy_corrections_ee_filtered * dy_event_weights_ee
total_new_dy_weight_ee = np.sum(new_dy_weights_ee_filtered)
print(f"Average new DY event weight in EE channel: {np.mean(new_dy_weights_ee_filtered):.6f}")
print(new_dy_weights_ee_filtered[:100])

scaling_factor_carl_ee = total_data_mc_weight_ee / total_new_dy_weight_ee
new_dy_weights_ee_filtered = new_dy_weights_ee_filtered * scaling_factor_carl_ee
new_dy_weights_ee = variable_list["event_weight"][1].copy()
new_dy_weights_ee[dy_mask_ee] = new_dy_weights_ee_filtered

print("----------- IN THE EE CHANNEL -----------")
print(f'Sum of original DY event weights: {np.sum(dy_event_weights_ee):.6f}')
print(f'Sum of new DY event weights: {total_new_dy_weight_ee:.6f}')
print(f'Target sum (data - other mc): {total_data_mc_weight_ee:.6f}')
print(f'Scaling factor applied to new DY weights: {scaling_factor_carl_ee:.6f}')

# ----------------------------------------------------------------------------------
# PREPARE DY WEIGHTS = 1 (scaled)

dy_weights_no_correction_filtered = np.ones_like(filter_events_by_channel(*variable_list["event_weight"], channel_id)[1])
total_dy_weight_no_correction = np.sum(dy_weights_no_correction_filtered)
scaling_factor_no_correction = total_data_mc_weight / total_dy_weight_no_correction
dy_weights_no_correction_filtered = dy_weights_no_correction_filtered * scaling_factor_no_correction
dy_weights_no_correction = variable_list["event_weight"][1].copy()
dy_weights_no_correction[dy_mask] = dy_weights_no_correction_filtered

# ------------------------------------------------------------------------------------
# PLOTTING

np.save("dy_event_weights_carl_mumu.npy", new_dy_weights)
print(f"Event weights from CARL method saved. Shape: {new_dy_weights.shape}")
np.save("dy_event_weights_carl_ee.npy", new_dy_weights_ee)
print(f"Event weights from CARL method in EE channel saved. Shape: {new_dy_weights_ee.shape}")

