from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import order as od
import os
import correctionlib
import math
import torch
from torch import nn
from typing import Sequence, Generator

from modules.hbt.hbt.config.analysis_hbt import analysis_hbt
from modules.hbt.modules.columnflow.columnflow.plotting.plot_functions_1d import plot_variable_stack  # noqa: E501
from modules.hbt.modules.columnflow.columnflow.hist_util import create_hist_from_variables, fill_hist  # noqa: E501

# --------------------------------------------------------------------------------------------------
# DATALOADER FOR BATCHING
class InMemoryDataLoader:

    def __init__(
        self,
        input: torch.Tensor | Sequence[torch.Tensor],
        batch_size: int | None = None,
        shuffle: bool = False,
        drop_last: bool = True,
    ) -> None:
        self.input = input
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Generator[torch.Tensor | Sequence[torch.Tensor], None, None]:
        # parse input type
        multi_input = isinstance(self.input, (list, tuple))
        if multi_input and not self.input:
            raise ValueError("input sequence is empty")

        # create extraction indices
        input_len = (self.input[0] if multi_input else self.input).shape[0]
        if not input_len:
            raise ValueError("input has zero length")
        indices = (torch.randperm if self.shuffle else torch.arange)(input_len, dtype=torch.int32)

        # determine number of batches
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        n_batches = int((math.floor if self.drop_last else math.ceil)(input_len / self.batch_size))

        # yield batches
        for i in range(n_batches):
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
            if multi_input:
                yield type(self.input)(inp[batch_indices] for inp in self.input)
            else:
                yield self.input[batch_indices]

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
# print(variables)

# load DY weight corrections from json file
dy_file = "/afs/desy.de/user/a/alvesand/public/dy_corrections.json.gz"
dy_correction = correctionlib.CorrectionSet.from_file(dy_file)
correction_set = dy_correction["dy_weight"]
syst = "nom"
era = year.replace("22", "2022").replace("_v14", "EE")
channel_id = 5  # mumu channel
# --------------------------------------------------------------------------------------------------

# PREPARE VARIABLES WITH BINNING INFO
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

n_jet = config_inst.variables.n.njets
jet1_pt = config_inst.variables.n.jet1_pt
n_btag_pnet = config_inst.variables.n.nbjets_pnet

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


# binnig for input variables
ll_pt_bin_tupples = [(i * 10, (i + 1) * 10) for i in range(20)]
ll_pt_bin_tupples[-1] = (190, np.inf)

n_jet_bin_tupples = [(i, i + 1) for i in range(2, 6)]
n_jet_bin_tupples[-1] = (5, np.inf) 

n_btag_pnet_bin_tupples = [(i, i + 1) for i in range(2, 6)]
n_btag_pnet_bin_tupples[-1] = (5, np.inf) 

bb_mass_bin_tupples = [(i * 20, (i + 1) * 20) for i in range(15)]
bb_mass_bin_tupples[-1] = (90, np.inf) 

bb_pt_bin_tupples = [(i * 10, (i + 1) * 10) for i in range(20)]
bb_pt_bin_tupples[-1] = (190, np.inf) 


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
    var_name = var_instance.name.replace("dilep", "ll").replace("dibjet", "bb").replace("njets", "n_jet").replace("nbjets", "n_btag")

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
        data_arr, dy_arr, mc_arr = filter_events_by_channel(data_arr, dy_arr, mc_arr, channel_id)  # noqa: E501
        data_weights, dy_weights, mc_weights = filter_events_by_channel(  # noqa: E501
            data_weights, dy_weights, mc_weights, channel_id
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

def bin_data(array_to_bin, weight_to_bin, bin_tupples):
    # turn the arrays [...] into a list of arrays [[...] , [...], ...] of len(list)) = len(bins)
    # where the first array in the list only contains the events that should fall in the first ll_pt bin, etc.

    binned_arrays = []
    binned_weights = []

    for bin_min, bin_max in bin_tupples:
        bin_mask = (array_to_bin >= bin_min) & (array_to_bin < bin_max)
        binned_arrays.append(array_to_bin[bin_mask])
        binned_weights.append(weight_to_bin[bin_mask])

    return binned_arrays, binned_weights

def calculate_dy_yield(var, bin_tupples):
    data_arr, dy_arr, mc_arr = filter_events_by_channel(*variable_list[var], channel_id)
    data_weights, dy_weights, mc_weights = filter_events_by_channel(*variable_list["event_weight"], channel_id)

    # bin the event weights arrays over ll_pt bins
    _, data_binned_weights = bin_data(data_arr, data_weights, bin_tupples)
    _, dy_binned_weights = bin_data(dy_arr, dy_weights, bin_tupples)
    _, mc_binned_weights = bin_data(mc_arr, mc_weights, bin_tupples)   

    # number of expected DY events in each bin by doing Data-MC
    expected_dy_yield = []
    for i in range(len(bin_tupples)):
        n_data = np.sum(data_binned_weights[i])
        n_mc = np.sum(mc_binned_weights[i])
        n_dy_expected = n_data - n_mc
        expected_dy_yield.append(n_dy_expected)
    expected_dy_yield = torch.tensor(expected_dy_yield, dtype=torch.float32)[:, None]

    # number of actual DY events in each bin
    actual_dy_yield = []
    for i in range(len(bin_tupples)):
        n_dy_actual = np.sum(dy_binned_weights[i])
        actual_dy_yield.append(n_dy_actual)
    actual_dy_yield = torch.tensor(actual_dy_yield, dtype=torch.float32)[:, None]

    return expected_dy_yield, actual_dy_yield

def calculate_bin_importance(expected_dy_yield, bin_tupples):
    bin_importance = [expected_dy_yield[i] / sum(expected_dy_yield) for i in range(len(bin_tupples))] 
    bin_importance = torch.tensor(bin_importance, dtype=torch.float32)
    bin_importance = bin_importance / bin_importance[0]
    return bin_importance
# --------------------------------------------------------------------------------------------------

# fill the variable list with arrays for each variable
variable_list = {}
for var in variables:
    data_arr = _concat_to_numpy(temp_storage["data"][var])
    dy_arr = _concat_to_numpy(temp_storage["dy"][var])
    mc_arr = _concat_to_numpy(temp_storage["mc"][var])

    variable_list[var] = [data_arr, dy_arr, mc_arr]

# read ll_pt and event weights from files
_, dy_ll_pt, _ = filter_events_by_channel(*variable_list["ll_pt"], channel_id) 
_, dy_n_jet, _ = filter_events_by_channel(*variable_list["n_jet"], channel_id) 
_, dy_n_btag_pnet, _ = filter_events_by_channel(*variable_list["n_btag_pnet"], channel_id)
_, dy_bb_pt, _ = filter_events_by_channel(*variable_list["bb_pt"], channel_id) 
_, dy_bb_mass, _ = filter_events_by_channel(*variable_list["bb_mass"], channel_id)
data_weights, dy_weights, mc_weights = filter_events_by_channel(*variable_list["event_weight"], channel_id) 

expected_dy_yield_ll_pt, actual_dy_yield_ll_pt = calculate_dy_yield("ll_pt", ll_pt_bin_tupples)
expected_dy_yield_n_jet, actual_dy_yield_n_jet = calculate_dy_yield("n_jet", n_jet_bin_tupples)
expected_dy_yield_n_btag_pnet, actual_dy_yield_n_btag_pnet = calculate_dy_yield("n_btag_pnet", n_btag_pnet_bin_tupples)
expected_dy_yield_bb_pt, actual_dy_yield_bb_pt = calculate_dy_yield("bb_pt", bb_pt_bin_tupples)
expected_dy_yield_bb_mass, actual_dy_yield_bb_mass = calculate_dy_yield("bb_mass", bb_mass_bin_tupples)

# TODO: calculate_bin_importance now takes torch tensors as input, maybe this will be an issue later?
# calculate bin importance for loss weighting
# TODO: discuss other possible definitions of bin importance
bin_importance_ll_pt = calculate_bin_importance(expected_dy_yield_ll_pt, ll_pt_bin_tupples)
bin_importance_n_jet = calculate_bin_importance(expected_dy_yield_n_jet, n_jet_bin_tupples)
bin_importance_n_btag_pnet = calculate_bin_importance(expected_dy_yield_n_btag_pnet, n_btag_pnet_bin_tupples) 
bin_importance_bb_pt = calculate_bin_importance(expected_dy_yield_bb_pt, bb_pt_bin_tupples)
bin_importance_bb_mass = calculate_bin_importance(expected_dy_yield_bb_mass, bb_mass_bin_tupples)

bin_importance_n_btag_pnet[0] = bin_importance_n_btag_pnet[0]*0.4 
bin_importance_n_jet[0] = bin_importance_n_jet[0]*0.4  # decrease importance of first n_jet bin
bin_importance_ll_pt[0] = bin_importance_ll_pt[0]*2  # increase importance of first ll_pt bin

# ----------------------------------------------------------------------------------
# SETUP for the NN
class DYReweightingNN(nn.Module):
    def __init__(self):
        super(DYReweightingNN, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # input layer to hidden layer
        self.bn1 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)  # hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

device = torch.device("cpu")
model = DYReweightingNN().to(device)
lr = 0.001
lr_threshold = 0.04
loss_fn = nn.MSELoss(reduction='none')
loss_target = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = 2048
n_epochs = 200

def calculate_loss(predicted, expected, bin_importance):
            return sum(
                loss_fn(predicted[i] / expected[i], torch.tensor([1.0])) * bin_importance[i] for i in range(len(predicted))
            ) / torch.sum(bin_importance)

original_loss_ll_pt = calculate_loss(actual_dy_yield_ll_pt, expected_dy_yield_ll_pt, bin_importance_ll_pt)
original_loss_n_jet = calculate_loss(actual_dy_yield_n_jet, expected_dy_yield_n_jet, bin_importance_n_jet)
original_loss_n_btag_pnet = calculate_loss(actual_dy_yield_n_btag_pnet, expected_dy_yield_n_btag_pnet, bin_importance_n_btag_pnet)
original_loss_bb_pt = calculate_loss(actual_dy_yield_bb_pt, expected_dy_yield_bb_pt, bin_importance_bb_pt)
original_loss_bb_mass = calculate_loss(actual_dy_yield_bb_mass, expected_dy_yield_bb_mass, bin_importance_bb_mass)

original_loss = sum(
    [original_loss_ll_pt,
    original_loss_n_jet,
    original_loss_n_btag_pnet,
    original_loss_bb_pt,
    original_loss_bb_mass]
)
print(f'Loss before training: {original_loss.item():.6f}\n')

# prepare dataloader
dy_ll_pt = torch.tensor(dy_ll_pt, dtype=torch.float)[:, None]
dy_n_jet = torch.tensor(dy_n_jet, dtype=torch.float)[:, None]
dy_n_btag_pnet = torch.tensor(dy_n_btag_pnet, dtype=torch.float)[:, None]
dy_bb_pt = torch.tensor(dy_bb_pt, dtype=torch.float)[:, None]
dy_bb_mass = torch.tensor(dy_bb_mass, dtype=torch.float)[:, None]
dy_weights = torch.tensor(dy_weights, dtype=torch.float)[:, None]
data_loader = InMemoryDataLoader([dy_ll_pt, dy_n_jet, dy_n_btag_pnet, dy_bb_pt, dy_bb_mass, dy_weights], batch_size=batch_size, shuffle=True)  # noqa: E501

# ----------------------------------------------------------------------------------

# TRAINING LOOP


# loop over batches
model.train()
dropped_lr = False

dy_ll_pt_mean, dy_ll_pt_std = dy_ll_pt.mean(), dy_ll_pt.std()
dy_n_jet_mean, dy_n_jet_std = dy_n_jet.mean(), dy_n_jet.std()
dy_n_btag_pnet_mean, dy_n_btag_pnet_std = dy_n_btag_pnet.mean(), dy_n_btag_pnet.std()
dy_bb_pt_mean, dy_bb_pt_std = dy_bb_pt.mean(), dy_bb_pt.std()
dy_bb_mass_mean, dy_bb_mass_std = dy_bb_mass.mean(), dy_bb_mass.std()

def get_batch_yields(array_batch, weights_batch, bin_tupples, total_len):
            _, weights_binned_in_var = bin_data(array_batch, weights_batch, bin_tupples)  
            dy_yield = [x.sum() * (total_len / len(array_batch)) for x in weights_binned_in_var]  # noqa: E501
            dy_yield = torch.stack(dy_yield)[:, None]
            return dy_yield


for epoch in range(n_epochs):
    for dy_ll_pt_batch, dy_n_jet_batch, dy_n_btag_pnet_batch, dy_bb_pt_batch, dy_bb_mass_batch, dy_weights_batch in data_loader:
        # reset gradients
        optimizer.zero_grad()

        # input normalization
        ll_pt_inputs = (dy_ll_pt_batch - dy_ll_pt_mean) / dy_ll_pt_std 
        n_jet_inputs = (dy_n_jet_batch - dy_n_jet_mean) / dy_n_jet_std  
        n_btag_pnet_inputs = (dy_n_btag_pnet_batch - dy_n_btag_pnet_mean) / dy_n_btag_pnet_std  
        bb_pt_inputs = (dy_bb_pt_batch - dy_bb_pt_mean) / dy_bb_pt_std  
        bb_mass_inputs = (dy_bb_mass_batch - dy_bb_mass_mean) / dy_bb_mass_std  
        inputs = torch.cat((ll_pt_inputs, n_jet_inputs, n_btag_pnet_inputs, bb_pt_inputs, bb_mass_inputs), dim=1) 

        # get dy weight predictions from the model
        pred_correction = model(inputs)

        # loss calculation
        updated_weights_batch = dy_weights_batch * pred_correction

        pred_dy_yield_ll_pt = get_batch_yields(dy_ll_pt_batch, updated_weights_batch, ll_pt_bin_tupples, len(dy_ll_pt))  # noqa: E501
        pred_dy_yield_n_jet = get_batch_yields(dy_n_jet_batch, updated_weights_batch, n_jet_bin_tupples, len(dy_n_jet))  # noqa: E501
        pred_dy_yield_n_btag_pnet = get_batch_yields(dy_n_btag_pnet_batch, updated_weights_batch, n_btag_pnet_bin_tupples, len(dy_n_btag_pnet))  # noqa: E501
        pred_dy_yield_bb_mass = get_batch_yields(dy_bb_mass_batch, updated_weights_batch, bb_mass_bin_tupples, len(dy_bb_mass))  # noqa: E501
        pred_dy_yield_bb_pt = get_batch_yields(dy_bb_pt_batch, updated_weights_batch, bb_pt_bin_tupples, len(dy_bb_pt))  # noqa

        loss_ll_pt = calculate_loss(pred_dy_yield_ll_pt, expected_dy_yield_ll_pt, bin_importance_ll_pt)
        loss_n_jet = calculate_loss(pred_dy_yield_n_jet, expected_dy_yield_n_jet, bin_importance_n_jet)
        loss_n_btag_pnet = calculate_loss(pred_dy_yield_n_btag_pnet, expected_dy_yield_n_btag_pnet, bin_importance_n_btag_pnet)
        loss_bb_mass = calculate_loss(pred_dy_yield_bb_mass, expected_dy_yield_bb_mass, bin_importance_bb_mass)
        loss_bb_pt = calculate_loss(pred_dy_yield_bb_pt, expected_dy_yield_bb_pt, bin_importance_bb_pt)
        
        loss = sum(
            [loss_ll_pt,
            loss_n_jet,
            loss_n_btag_pnet,
            loss_bb_mass,
            loss_bb_pt]
        )


        # backpropagation and optimization steps
        loss.backward()
        optimizer.step()

    print(f"---- Epoch {epoch + 1}/{n_epochs} : Loss = {loss.item():.6f}, mean = {pred_correction.mean().item():.6f}, std = {pred_correction.std().item():.6f}")  # noqa: E501

    if (dropped_lr is False) and (loss.item() <= lr_threshold):
        learn_rate = lr * 0.5  # decrease learning for later epochs
        dropped_lr = True
        print(f"-> Learning rate decreased from {lr} to {learn_rate}!")
        for param_group in optimizer.param_groups:
            param_group['lr'] = learn_rate

    if loss.item() <= loss_target:
        print("-> Early stopping criterion met. Stopping training. ")
        break

print("\n---------------- Training completed -----------------")
# print("\nFinal ratios pred_dy_yield / expected_dy_yield :")
# print(pred_dy_yield_ll_pt/expected_dy_yield_ll_pt)
# print(pred_dy_yield_n_jet/expected_dy_yield_n_jet)
# print(pred_dy_yield_n_btag_pnet/expected_dy_yield_n_btag_pnet)
# print(pred_dy_yield_bb_mass/expected_dy_yield_bb_mass)
# print(pred_dy_yield_bb_pt/expected_dy_yield_bb_pt)

# ----------------------------------------------------------------------------------

# PLOTTING RESULTS

model.eval()
with torch.no_grad():
    ll_pt_inputs = (dy_ll_pt - dy_ll_pt_mean) / dy_ll_pt_std
    n_jet_inputs = (dy_n_jet - dy_n_jet_mean) / dy_n_jet_std
    n_btag_pnet_inputs = (dy_n_btag_pnet - dy_n_btag_pnet_mean) / dy_n_btag_pnet_std
    bb_pt_inputs = (dy_bb_pt - dy_bb_pt_mean) / dy_bb_pt_std
    bb_mass_inputs = (dy_bb_mass - dy_bb_mass_mean) / dy_bb_mass_std
    inputs = torch.cat((ll_pt_inputs, n_jet_inputs, n_btag_pnet_inputs, bb_pt_inputs, bb_mass_inputs), dim=1)
    pred_correction = model(inputs).squeeze(1)

final_weights = dy_weights.flatten() * pred_correction.detach().cpu().numpy().flatten()

plot_function(bb_eta, "nn_with_batching_weights_bb_eta", dy_weights=final_weights)
plot_function(bb_mass, "nn_with_batching_weights_bb_mass", dy_weights=final_weights)
plot_function(bb_phi, "nn_with_batching_weights_bb_phi", dy_weights=final_weights)
plot_function(bb_pt, "nn_with_batching_weights_bb_pt", dy_weights=final_weights)
plot_function(jet1_pt, "nn_with_batching_weights_jet1_pt", dy_weights=final_weights)
plot_function(ll_eta, "nn_with_batching_weights_ll_eta", dy_weights=final_weights)
plot_function(ll_mass, "nn_with_batching_weights_ll_mass", dy_weights=final_weights)
plot_function(ll_phi, "nn_with_batching_weights_ll_phi", dy_weights=final_weights)
plot_function(ll_pt, "nn_with_batching_weights_ll_pt", dy_weights=final_weights)
plot_function(n_btag_pnet, "nn_with_batching_weights_n_btag_pnet", dy_weights=final_weights)
plot_function(n_jet, "nn_with_batching_weights_n_jet", dy_weights=final_weights)

