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
# SETUP of files
# choose which year to use
year = "22pre_v14"
channel_id = 5  # 4: ee channel, 5: mumu channel
# define file path
config_inst = analysis_hbt.get_config(year)
mypath = f"/data/dust/user/riegerma/hh2bbtautau/dy_dnn_data/inputs_prod20_vbf/{year}/"  # noqa: E501
filelist = os.listdir(mypath)
example_file = "w_lnu_1j_pt100to200_amcatnlo.parquet"
fullpath = f"{mypath}{example_file}"
example_array = ak.from_parquet(fullpath)
variables = example_array.fields
#print(variables)

# load DY weight corrections from json file
dy_file = "/afs/desy.de/user/a/alvesand/public/dy_corrections.json.gz"
dy_correction = correctionlib.CorrectionSet.from_file(dy_file)
correction_set = dy_correction["dy_weight"]
syst = "nom"
era = year.replace("22", "2022").replace("_v14", "EE")

# PREPARE VARIABLES AND BINNING INFO
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

met_pt = config_inst.variables.n.met_pt
met_phi = config_inst.variables.n.met_phi
n_btag_pnet = config_inst.variables.n.nbjets_pnet

# define binning
ll_pt_bin_tupples = [(i * 10, (i + 1) * 10) for i in range(20)]
ll_pt_bin_tupples[-1] = (190, np.inf) # overwrite last bin to consider overflow

n_jet_bin_tupples = [(i, i + 1) for i in range(2, 6)]
n_jet_bin_tupples[-1] = (5, np.inf) 

n_btag_pnet_bin_tupples = [(i, i + 1) for i in range(2, 6)]
n_btag_pnet_bin_tupples[-1] = (5, np.inf) 

bb_mass_bin_tupples = [(i * 20, (i + 1) * 20) for i in range(15)]
bb_mass_bin_tupples[-1] = (90, np.inf) 

bb_pt_bin_tupples = [(i * 10, (i + 1) * 10) for i in range(20)]
bb_pt_bin_tupples[-1] = (190, np.inf) 

# get processes
bkg_process = od.Process(name="mc", id="+", color1="#e76300", label="MC")
dy_process = config_inst.get_process("dy")
data_process = config_inst.get_process("data")


# helper to safely concatenate awkward arrays and return a numpy array
def _concat_to_numpy(lst):
        if not lst:
            return np.array([], dtype=float)
        concatenated = ak.concatenate(lst)
        try:
            return ak.to_numpy(concatenated)
        except Exception:
            return np.asarray(ak.to_list(concatenated), dtype=float)
        

# CREATE A DICTIONARY WITH ARRAYS FOR ALL VARIABLES
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

variable_list = {}
# fill the variable list with arrays for each variable
for var in variables: 
        data_arr = _concat_to_numpy(temp_storage["data"][var])
        dy_arr = _concat_to_numpy(temp_storage["dy"][var])
        mc_arr = _concat_to_numpy(temp_storage["mc"][var])
        
        variable_list[var] = [data_arr, dy_arr, mc_arr]
print("Created dictionary with arrays for all variables.")


# --------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS


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


def plot_function(var_instance: od.Variable, file_version: str, dy_weights=None, filter_events: bool = True, channel_id = None):  # noqa: E501
    var_name = var_instance.name.replace("dilep", "ll").replace("dibjet", "bb").replace("njets", "n_jet").replace("nbjets", "n_btag")  # noqa: E501

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
        binned_arrays.append( array_to_bin[bin_mask] )
        binned_weights.append( weight_to_bin[bin_mask] )

    return binned_arrays, binned_weights

def calculate_dy_yield(var, bin_tupples):
    data_arr, dy_arr, mc_arr = filter_events_by_channel(*variable_list[var], channel_id)
    data_weights, dy_weights, mc_weights = filter_events_by_channel(*variable_list["event_weight"], channel_id)

    # bin the event weights arrays over ll_pt bins
    _, data_binned_weights = bin_data(data_arr, data_weights, bin_tupples)
    _, dy_binned_weights = bin_data(dy_arr, dy_weights, bin_tupples)
    _, mc_binned_weights = bin_data(mc_arr, mc_weights, bin_tupples)

    # calculate expected and actual DY yield in each bin
    expected_dy_yield = []
    actual_dy_yield = []
    for i in range(len(bin_tupples)):
        n_data = np.sum(data_binned_weights[i])
        n_mc = np.sum(mc_binned_weights[i])
        n_dy_expected = n_data - n_mc
        expected_dy_yield.append(n_dy_expected)

        n_dy_actual = np.sum(dy_binned_weights[i])
        actual_dy_yield.append(n_dy_actual)

    expected_dy_yield = torch.tensor(expected_dy_yield, dtype=torch.float32)[:, None]
    actual_dy_yield = torch.tensor(actual_dy_yield, dtype=torch.float32)[:, None]

    return expected_dy_yield, actual_dy_yield

def calculate_bin_importance(expected_dy_yield, bin_tupples):
    bin_importance = [expected_dy_yield[i] / sum(expected_dy_yield) for i in range(len(bin_tupples))]
    bin_importance = [imp / bin_importance[0] for imp in bin_importance]  

    return torch.tensor(bin_importance, dtype=torch.float32)

def get_bin_indices(var_array, bin_tupples):
    # initialize with -1 (meaning: not in any bin)
    bin_indices = torch.full_like(var_array, -1, dtype=torch.long)
    for i, (bin_min, bin_max) in enumerate(bin_tupples):
        mask = (bin_min <= var_array) & (var_array < bin_max)
        bin_indices[mask] = i
    # returns a tensor of same length as var_array with the bin index for each event
    return bin_indices

def calculate_yields_fast(bin_idx, weights, bin_tupples):
    dy_yield = torch.zeros(len(bin_tupples), dtype=torch.float32)
    mask = bin_idx != -1 # only consider events that fall into a bin
    dy_yield.scatter_add_(0, bin_idx[mask].squeeze(), weights[mask].squeeze())
    return dy_yield

# --------------------------------------------------------------------------------------------------
# BINNING AND DATA PREPARATION FOR TRAINING


# read n_jet, ll_pt and event weights from files
_, dy_n_jet, _ = filter_events_by_channel(*variable_list["n_jet"], channel_id)
_, dy_ll_pt, _ = filter_events_by_channel(*variable_list["ll_pt"], channel_id)
_, dy_n_btag_pnet, _ = filter_events_by_channel(*variable_list["n_btag_pnet"], channel_id)
_, dy_bb_mass, _ = filter_events_by_channel(*variable_list["bb_mass"], channel_id)
_, dy_bb_pt, _ = filter_events_by_channel(*variable_list["bb_pt"], channel_id)
_, dy_weights, _ = filter_events_by_channel(*variable_list["event_weight"], channel_id)

dy_ll_pt = torch.tensor(dy_ll_pt, dtype=torch.float32)[:, None]
dy_n_jet = torch.tensor(dy_n_jet, dtype=torch.float32)[:, None]
dy_n_btag_pnet = torch.tensor(dy_n_btag_pnet, dtype=torch.float32)[:, None]
dy_bb_mass = torch.tensor(dy_bb_mass, dtype=torch.float32)[:, None]
dy_bb_pt = torch.tensor(dy_bb_pt, dtype=torch.float32)[:, None]
dy_weights = torch.tensor(dy_weights, dtype=torch.float32)[:, None]


expected_dy_ll_pt_yield, actual_dy_ll_pt_yield = calculate_dy_yield("ll_pt", ll_pt_bin_tupples)
expected_dy_n_jet_yield, actual_dy_n_jet_yield = calculate_dy_yield("n_jet", n_jet_bin_tupples)
expected_dy_n_btag_pnet_yield, actual_dy_n_btag_pnet_yield = calculate_dy_yield("n_btag_pnet", n_btag_pnet_bin_tupples)
expected_dy_bb_mass_yield, actual_dy_bb_mass_yield = calculate_dy_yield("bb_mass", bb_mass_bin_tupples)
expected_dy_bb_pt_yield, actual_dy_bb_pt_yield = calculate_dy_yield("bb_pt", bb_pt_bin_tupples)

# TODO: calculate_bin_importance now takes torch tensors as input, maybe this will be an issue later?
# calculate bin importance for loss weighting
# TODO: discuss other possible definitions of bin importance
bin_importance_ll_pt = calculate_bin_importance(expected_dy_ll_pt_yield, ll_pt_bin_tupples)
bin_importance_n_jet = calculate_bin_importance(expected_dy_n_jet_yield, n_jet_bin_tupples)
bin_importance_n_btag_pnet = calculate_bin_importance(expected_dy_n_btag_pnet_yield, n_btag_pnet_bin_tupples)
bin_importance_bb_mass = calculate_bin_importance(expected_dy_bb_mass_yield, bb_mass_bin_tupples)
bin_importance_bb_pt = calculate_bin_importance(expected_dy_bb_pt_yield, bb_pt_bin_tupples)

bin_importance_n_btag_pnet[0] = bin_importance_n_btag_pnet[0] * 0.4
bin_importance_n_jet[0] = bin_importance_n_jet[0] * 0.4
bin_importance_ll_pt[0] = bin_importance_ll_pt[0] * 2.0

bin_idx_ll_pt = get_bin_indices(dy_ll_pt, ll_pt_bin_tupples)
bin_idx_n_jet = get_bin_indices(dy_n_jet, n_jet_bin_tupples)
bin_idx_n_btag_pnet = get_bin_indices(dy_n_btag_pnet, n_btag_pnet_bin_tupples)
bin_idx_bb_mass = get_bin_indices(dy_bb_mass, bb_mass_bin_tupples)
bin_idx_bb_pt = get_bin_indices(dy_bb_pt, bb_pt_bin_tupples)

#-------------------------------------------------------------------------------------------------
# SETUP FOR THE THE NN
# use a simple feedforward NN with 5-10-10-10-1 architecture, ReLu activation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DYReweightingNN(nn.Module):
    def __init__(self):
        super(DYReweightingNN, self).__init__()
        self.fc1 = nn.Linear(5, 10)   # input layer to hidden layer
        self.bn1 = nn.BatchNorm1d(10) # normalize inputs to hidden layer for more efficient training
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)  # hidden layer to hidden layer
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 10)  # hidden layer to hidden layer
        self.bn3 = nn.BatchNorm1d(10)  
        self.fc4 = nn.Linear(10, 1)   # hidden layer to output

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

model = DYReweightingNN().to(device)
lr = 0.005
lr_threshold = 0.2
loss_target = 0.01
dropped_lr = False
loss_fn = nn.MSELoss(reduction = 'none')  # we will do custom reduction for bin importance weighting
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
n_epochs = 350

def calculate_loss(predicted, expected, bin_importance):
            ratios = predicted.flatten() / expected.flatten()
            return torch.sum(
                loss_fn(ratios, torch.ones_like(ratios)) * bin_importance.flatten()
            )/ torch.sum(bin_importance.flatten())

original_loss_ll_pt = calculate_loss(actual_dy_ll_pt_yield, expected_dy_ll_pt_yield, bin_importance_ll_pt)
original_loss_n_jet = calculate_loss(actual_dy_n_jet_yield, expected_dy_n_jet_yield, bin_importance_n_jet)
original_loss_n_btag_pnet = calculate_loss(actual_dy_n_btag_pnet_yield, expected_dy_n_btag_pnet_yield, bin_importance_n_btag_pnet)
original_loss_bb_mass = calculate_loss(actual_dy_bb_mass_yield, expected_dy_bb_mass_yield, bin_importance_bb_mass)
original_loss_bb_pt = calculate_loss(actual_dy_bb_pt_yield, expected_dy_bb_pt_yield, bin_importance_bb_pt)

original_loss = sum(
    [original_loss_ll_pt, 
    original_loss_n_jet, 
    original_loss_n_btag_pnet, 
    original_loss_bb_mass, 
    original_loss_bb_pt]
)
print(f'Loss before training: {original_loss.item():.6f}')


ll_pt_inputs = (dy_ll_pt - dy_ll_pt.mean()) / dy_ll_pt.std()
n_jet_inputs = (dy_n_jet - dy_n_jet.mean()) / dy_n_jet.std()
n_btag_pnet_inputs = (dy_n_btag_pnet - dy_n_btag_pnet.mean()) / dy_n_btag_pnet.std()
bb_mass_inputs = (dy_bb_mass - dy_bb_mass.mean()) / dy_bb_mass.std()
bb_pt_inputs = (dy_bb_pt - dy_bb_pt.mean()) / dy_bb_pt.std()
inputs = torch.cat(
    (ll_pt_inputs, n_jet_inputs, n_btag_pnet_inputs, bb_mass_inputs, bb_pt_inputs), 
    dim=1
)

#-------------------------------------------------------------------------------------------------

# TRAINING LOOP

model.train()

for epoch in range(n_epochs):

    optimizer.zero_grad()

    pred_correction = model(inputs)

    updated_weights = dy_weights * pred_correction

    pred_dy_yield_ll_pt = calculate_yields_fast(bin_idx_ll_pt, updated_weights, ll_pt_bin_tupples)
    pred_dy_yield_n_jet = calculate_yields_fast(bin_idx_n_jet, updated_weights, n_jet_bin_tupples)
    pred_dy_yield_n_btag_pnet = calculate_yields_fast(bin_idx_n_btag_pnet, updated_weights, n_btag_pnet_bin_tupples)
    pred_dy_yield_bb_mass = calculate_yields_fast(bin_idx_bb_mass, updated_weights, bb_mass_bin_tupples)
    pred_dy_yield_bb_pt = calculate_yields_fast(bin_idx_bb_pt, updated_weights, bb_pt_bin_tupples)

    # define loss = (loss_bin1 + loss_bin2 + ...)/sum(bin_importance)
    loss_ll_pt = calculate_loss(pred_dy_yield_ll_pt, expected_dy_ll_pt_yield, bin_importance_ll_pt) 
    loss_n_jet = calculate_loss(pred_dy_yield_n_jet, expected_dy_n_jet_yield, bin_importance_n_jet)
    loss_n_btag_pnet = calculate_loss(pred_dy_yield_n_btag_pnet, expected_dy_n_btag_pnet_yield, bin_importance_n_btag_pnet)
    loss_bb_mass = calculate_loss(pred_dy_yield_bb_mass, expected_dy_bb_mass_yield, bin_importance_bb_mass)
    loss_bb_pt = calculate_loss(pred_dy_yield_bb_pt, expected_dy_bb_pt_yield, bin_importance_bb_pt)

    loss = sum(
        [loss_ll_pt, 
        loss_n_jet, 
        loss_n_btag_pnet, 
        loss_bb_mass, 
        loss_bb_pt]
    )

    # do backpropagation and optimization step
    loss.backward()
    optimizer.step()

    print(f"---- Epoch {epoch+1}/{n_epochs}: Loss = {loss.item():.6f}, mean = {pred_correction.mean().item():.6f}, std = {pred_correction.std().item():.6f}")

    if (dropped_lr is False) and (loss.item() <= lr_threshold):
        learn_rate = lr * 0.5  # decrease learning for later epochs
        dropped_lr = True
        print(f"-> Learning rate decreased from {lr} to {learn_rate}!")
        for param_group in optimizer.param_groups:
            param_group['lr'] = learn_rate

    if loss.item() <= loss_target:
        print("-> Early stopping criterion met. Stopping training. ")
        break

print(f"\n---------------- Training completed after {epoch+1} epochs-----------------")
print(f"Final Loss: {loss.item():.6f}")
# print("\nFinal ratios pred_dy_yield / expected_dy_yield :")
# print(pred_dy_yield/expected_dy_yield)

# ----------------------------------------------------------------------------------

# EVALUATION AND PLOTTING

model.eval()
with torch.no_grad():
    pred_correction = model(inputs).squeeze(1)

_, _, _, _, dy_mask, _ = filter_events_by_channel(
    variable_list["ll_pt"][0],
    variable_list["ll_pt"][1],
    variable_list["ll_pt"][2],
    channel_id,
    return_masks=True,
)
final_weights = variable_list["event_weight"][1].copy()
final_weights[dy_mask] = final_weights[dy_mask] * pred_correction.detach().cpu().numpy().flatten()

plot_function(ll_pt, "nn_weights_ll_pt", dy_weights=final_weights, channel_id=channel_id)
plot_function(ll_mass, "nn_weights_ll_mass", dy_weights=final_weights, channel_id=channel_id)
plot_function(ll_eta, "nn_weights_ll_eta", dy_weights=final_weights, channel_id=channel_id)
plot_function(ll_phi, "nn_weights_ll_phi", dy_weights=final_weights, channel_id=channel_id)

plot_function(bb_pt, "nn_weights_bb_pt", dy_weights=final_weights, channel_id=channel_id)
plot_function(bb_mass, "nn_weights_bb_mass", dy_weights=final_weights, channel_id=channel_id)
plot_function(bb_eta, "nn_weights_bb_eta", dy_weights=final_weights, channel_id=channel_id)
plot_function(bb_phi, "nn_weights_bb_phi", dy_weights=final_weights, channel_id=channel_id)

plot_function(jet1_pt, "nn_weights_jet1_pt", dy_weights=final_weights, channel_id=channel_id)
plot_function(n_jet, "nn_weights_n_jet", dy_weights=final_weights, channel_id=channel_id)
plot_function(n_btag_pnet, "nn_weights_n_btag_pnet", dy_weights=final_weights, channel_id=channel_id)

# from IPython import embed; embed()

# plt.scatter(dy_ll_pt.numpy(), pred_correction.detach().numpy())
# plt.xlabel("ll_pt [GeV]")
# plt.ylabel("predicted DY weight correction")
# plt.title("DY NN Weight Correction vs ll_pt")
# plt.savefig("scatter_plot_ll_pt_correction.pdf")