import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import order as od
import os
import correctionlib

import torch
from torch import nn

from modules.hbt.hbt.config.analysis_hbt import analysis_hbt
from modules.hbt.modules.columnflow.columnflow.plotting.plot_functions_1d import plot_variable_stack  # noqa: E501
from modules.hbt.modules.columnflow.columnflow.hist_util import create_hist_from_variables, fill_hist  # noqa: E501

# --------------------------------------------------------------------------------------------------
# FILES SETUP
year = "22pre_v14"
config_inst = analysis_hbt.get_config(year)
mypath = f"/data/dust/user/riegerma/hh2bbtautau/dy_dnn_data/inputs_prod20_vbf/{year}/"  # noqa: E501
filelist = os.listdir(mypath)
example_file = "w_lnu_1j_pt100to200_amcatnlo.parquet"
fullpath = f"{mypath}{example_file}"
example_array = ak.from_parquet(fullpath)
variables = example_array.fields
# print(variables)

# load DY weight corrections from json file
dy_file = "/afs/desy.de/user/a/alvesand/public/dy_corrections.json.gz"
dy_correction = correctionlib.CorrectionSet.from_file(dy_file)
correction_set = dy_correction["dy_weight"]
syst = "nom"
era = year.replace("22", "2022").replace("_v14", "EE")
# --------------------------------------------------------------------------------------------------

# get variables with binning info
ll_mass = config_inst.variables.n.dilep_mass
ll_pt = config_inst.variables.n.dilep_pt
ll_eta = config_inst.variables.n.dilep_eta
ll_phi = config_inst.variables.n.dilep_phi

bb_mass = config_inst.variables.n.dibjet_mass
bb_pt = config_inst.variables.n.dibjet_pt
bb_eta = config_inst.variables.n.dibjet_eta
bb_phi = config_inst.variables.n.dibjet_phi

hh_mass = config_inst.variables.n.hh_mass
hh_pt = config_inst.variables.n.hh_pt
hh_eta = config_inst.variables.n.hh_eta
hh_phi = config_inst.variables.n.hh_phi
jet1_pt = config_inst.variables.n.jet1_pt

met_pt = config_inst.variables.n.met_pt
met_phi = config_inst.variables.n.met_phi

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


# binnig
bin_tupples = [(i * 10, (i + 1) * 10) for i in range(20)]
bin_tupples[-1] = (190, np.inf)

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


def filter_events_by_channel(data_arr, dy_arr, mc_arr, desired_channel_id, return_masks=False):
    """
    Function to filter the events by channel id and DY region cuts
    -> desired_channel_id: ee = 4, mumu = 5
    """
    data_channel_id, dy_channel_id, mc_channel_id = variable_list["channel_id"]  # noqa: E501
    data_ll_mass, dy_ll_mass, mc_ll_mass = variable_list["ll_mass"]  # noqa: E501
    data_met, dy_met, mc_met = variable_list["met_pt"]

    # create masks
    data_id_mask = data_channel_id == desired_channel_id
    dy_id_mask = dy_channel_id == desired_channel_id
    mc_id_mask = mc_channel_id == desired_channel_id

    data_ll_mask = (data_ll_mass >= 70) & (data_ll_mass <= 110)
    dy_ll_mask = (dy_ll_mass >= 70) & (dy_ll_mass <= 110)
    mc_ll_mask = (mc_ll_mass >= 70) & (mc_ll_mass <= 110)

    data_met_mask = data_met < 45
    dy_met_mask = dy_met < 45
    mc_met_mask = mc_met < 45

    # combine masks
    data_mask = data_id_mask & data_ll_mask & data_met_mask
    dy_mask = dy_id_mask & dy_ll_mask & dy_met_mask
    mc_mask = mc_id_mask & mc_ll_mask & mc_met_mask

    if return_masks:
        return data_arr[data_mask], dy_arr[dy_mask], mc_arr[mc_mask], data_mask, dy_mask, mc_mask
    else:
        return data_arr[data_mask], dy_arr[dy_mask], mc_arr[mc_mask]


def hist_function(var_instance, data, weights):
    h = create_hist_from_variables(var_instance, weight=True)
    fill_hist(h, {var_instance.name: data, "weight": weights})
    return h


def plot_function(var_instance: od.Variable, file_version: str, dy_weights=None, filter_events: bool = True):  # noqa: E501
    var_name = var_instance.name.replace("dilep", "ll").replace("dibjet", "bb")

    # get variable arrays and original event weights from files
    data_arr, dy_arr, mc_arr = variable_list[var_name]
    data_weights, dy_weights_original, mc_weights = variable_list["event_weight"]  # noqa: E501

    # use original DY weights if none are provided
    if dy_weights is None:
        print("--> Using original DY event weights!")
        dy_weights = dy_weights_original
    else:
        print("--> Using updated DY event weights!")
        dy_weights = dy_weights

    # filter arrays considering mumu channel id and DY region cuts
    if filter_events:
        data_arr, dy_arr, mc_arr = filter_events_by_channel(data_arr, dy_arr, mc_arr, 5)  # noqa: E501
        data_weights, dy_weights, mc_weights = filter_events_by_channel(  # noqa: E501
            data_weights, dy_weights, mc_weights, 5
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


def bin_data(array_to_bin, weight_to_bin, bins=bin_tupples):
    # turn the arrays [...] into a list of arrays [[...] , [...], ...] of len(list)) = len(bins)
    # where the first array in the list only contains the events that should fall in the first pt_ll bin, etc.

    binned_arrays = []
    binned_weights = []

    for bin_min, bin_max in bins:
        bin_mask = (array_to_bin >= bin_min) & (array_to_bin < bin_max)
        binned_arrays.append(array_to_bin[bin_mask])
        binned_weights.append(weight_to_bin[bin_mask])

    return binned_arrays, binned_weights


# fill the variable list with arrays for each variable
variable_list = {}
for var in variables:
    data_arr = _concat_to_numpy(temp_storage["data"][var])
    dy_arr = _concat_to_numpy(temp_storage["dy"][var])
    mc_arr = _concat_to_numpy(temp_storage["mc"][var])

    variable_list[var] = [data_arr, dy_arr, mc_arr]

# read pt_ll and event weights from files
data_pt_ll, dy_pt_ll, mc_pt_ll = filter_events_by_channel(*variable_list["ll_pt"], 5)  # noqa: E501
data_weights, dy_weights, mc_weights = filter_events_by_channel(*variable_list["event_weight"], 5)  # noqa: E501

# bin the event weights arrays over pt_ll bins
_, data_binned_weights = bin_data(data_pt_ll, data_weights)
dy_binned_pt_ll, dy_binned_weights = bin_data(dy_pt_ll, dy_weights)
_, mc_binned_weights = bin_data(mc_pt_ll, mc_weights)

# number of expected DY events in each bin by doing Data-MC
expected_dy_yield = []
for i in range(len(bin_tupples)):
    n_data = np.sum(data_binned_weights[i])
    n_mc = np.sum(mc_binned_weights[i])
    n_dy_expected = n_data - n_mc
    expected_dy_yield.append(n_dy_expected)
# print(f'Expected DY events per bin: {expected_dy_yield}')

# number of actual DY events in each bin
actual_dy_yield = []
for i in range(len(bin_tupples)):
    n_dy_actual = np.sum(dy_binned_weights[i])
    actual_dy_yield.append(n_dy_actual)
# print(f'Actual DY events per bin: {actual_dy_yield}')

# calculate bin importance for loss weighting
# TODO: discuss other possible definitions of bin importance
bin_importance = [sum(data_binned_weights[i]) / sum(data_weights) for i in range(len(bin_tupples))]  # noqa: E501
bin_importance = torch.tensor(bin_importance, dtype=torch.float32)
bin_importance = bin_importance / bin_importance[0]
print(f'Bin importance values: {bin_importance}')


# ----------------------------------------------------------------------------------
# NN SETUP

# use a simple feedforward NN with 1-10-1 architecture, ReLu activation
class DYReweightingNN(nn.Module):
    def __init__(self):
        super(DYReweightingNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # input layer to hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)  # hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = DYReweightingNN().to(device)
lr = 0.0015
loss_fn = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = int(len(dy_pt_ll)/2)
n_epochs = 200
# ----------------------------------------------------------------------------------

# convert actual and expected dy yields to torch tensors
actual_dy_yield = torch.tensor(actual_dy_yield, dtype=torch.float32)[:, None]
expected_dy_yield = torch.tensor(expected_dy_yield, dtype=torch.float32)[:, None]  # noqa: E501

# calculate and print original loss before training
original_loss = (sum(
    loss_fn(actual_dy_yield[i] / expected_dy_yield[i], torch.tensor([1.0])) * bin_importance[i] for i in range(len(bin_tupples))  # noqa: E501
    ) / torch.sum(bin_importance)
)
print(f'Loss before training: {original_loss.item():.6f}\n')

# prepare dataloader
dy_pt_ll = torch.tensor(dy_pt_ll, dtype=torch.float)[:, None]
dy_weights = torch.tensor(dy_weights, dtype=torch.float)[:, None]
dataset = torch.utils.data.TensorDataset(dy_pt_ll, dy_weights)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)  # noqa: E501

# TODO : fix data_loader

# loop over batches
for epoch in range(n_epochs):
    for i, (inputs_pt_ll, weights) in enumerate(data_loader):
        inputs_pt_ll = dy_pt_ll
        weights = dy_weights
        optimizer.zero_grad()

        # normalize inputs
        inputs = (inputs_pt_ll - inputs_pt_ll.mean()) / inputs_pt_ll.std()

        # get dy weight predictions from the model
        pred_correction = model(inputs)
        _, batched_pred_correction_binned = bin_data(inputs_pt_ll, pred_correction)  # noqa: E501
        _, batched_weights_binned = bin_data(inputs_pt_ll, weights)

        # update event weights with predicted correction
        updated_weights = [x*y for x, y in zip(batched_pred_correction_binned, batched_weights_binned)]  # noqa: E501
        pred_dy_yield = [x.sum() * (len(dy_pt_ll) / len(inputs_pt_ll)) for x in updated_weights]  # noqa: E501
        pred_dy_yield = torch.stack(pred_dy_yield)[:, None]
        loss = torch.sum(loss_fn(pred_dy_yield/expected_dy_yield, torch.tensor(1.0)) * bin_importance[:, None], dim=0) / torch.sum(bin_importance)  # noqa: E501

        # do backpropagation and optimization step
        loss.backward()
        optimizer.step()

    print(f"---- Epoch {epoch + 1}/{n_epochs} : Loss = {loss.item():.6f}, mean = {pred_correction.mean().item():.6f}, std = {pred_correction.std().item():.6f}")  # noqa: E501

print("\n---------------- Training completed -----------------")
print("\nFinal ratios pred_dy_yield / expected_dy_yield :")
print(pred_dy_yield/expected_dy_yield)

inputs = (dy_pt_ll - dy_pt_ll.mean()) / dy_pt_ll.std()
pred_correction = model(inputs).squeeze(1)
_, _, _, _, dy_mask, _ = filter_events_by_channel(
    variable_list["ll_pt"][0],
    variable_list["ll_pt"][1],
    variable_list["ll_pt"][2],
    5,
    return_masks=True,
)
final_weights = variable_list["event_weight"][1]
final_weights[dy_mask] = final_weights[dy_mask] * pred_correction.detach().cpu().numpy().flatten()
plot_function(ll_pt, "nn_weights_ll_pt", dy_weights=final_weights)
plot_function(ll_mass, "nn_weights_ll_mass", dy_weights=final_weights)
plot_function(ll_eta, "nn_weights_ll_eta", dy_weights=final_weights)
plot_function(bb_pt, "nn_weights_bb_pt", dy_weights=final_weights)
plot_function(bb_mass, "nn_weights_bb_mass", dy_weights=final_weights)
plot_function(bb_eta, "nn_weights_bb_eta", dy_weights=final_weights)
plot_function(jet1_pt, "nn_weights_jet1_pt", dy_weights=final_weights)
