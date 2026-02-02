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
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader

from modules.hbt.hbt.config.analysis_hbt import analysis_hbt
from modules.hbt.modules.columnflow.columnflow.plotting.plot_functions_1d import plot_variable_stack  # noqa: E501
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
# print(variables)

# load DY weight corrections from json file
dy_file = "/afs/desy.de/user/a/alvesand/public/dy_corrections.json.gz"
dy_correction = correctionlib.CorrectionSet.from_file(dy_file)
correction_set = dy_correction["dy_weight"]
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

def plot_score_distribution(model, loader, file_name: str):
    model.eval()
    loss = 0.0
    all_scores = []
    all_labels = []
    all_weights = []

    with torch.no_grad():
        for inputs, labels, weights in loader:
            outputs = model(inputs)
            batch_loss = torch.mean(loss_fn(outputs, labels).flatten()*weights.flatten())
            loss += batch_loss.item()
            # save results for analysis
            all_scores.append(outputs)
            all_labels.append(labels)
            all_weights.append(weights)

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_weights = torch.cat(all_weights).numpy()

    data_scores = all_scores[all_labels == 1]
    dy_scores = all_scores[all_labels == 0]
    data_weights = all_weights[all_labels == 1]
    dy_weights = all_weights[all_labels == 0]

    plt.figure(figsize=(8,6))
    plt.hist(data_scores, bins=50, weights=data_weights, density=True, alpha=0.5, label='Data', color='blue')  # noqa: E501
    plt.hist(dy_scores, bins=50, weights=dy_weights, density=True, alpha=0.5, label='Drell Yan', color='orange')  # noqa: E501
    plt.xlabel('Classifier Output')
    plt.ylabel('Normalized Events')
    plt.title('DY Classifier Output Distribution')
    plt.legend()
    plt.savefig(f"classifier_output_distribution_{file_name}.pdf")

    print(f"Saved DY classifier output distribution with loss = {loss/len(loader):.6f}")

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

for var in input_variables:
    data_var, dy_var, _ = filter_events_by_channel(*variable_list[var], channel_id)
    data_variables.append(torch.tensor(data_var, dtype=torch.float)[:, None])
    dy_variables.append(torch.tensor(dy_var, dtype=torch.float)[:, None])


dy_inputs = torch.cat(dy_variables, dim=1)
dy_labels = torch.zeros(len(dy_inputs), 1)
_, dy_weights, _ = filter_events_by_channel(*variable_list["event_weight"], channel_id) 
dy_weights = torch.tensor(dy_weights, dtype=torch.float)[:, None]

data_inputs = torch.cat(data_variables, dim=1)
data_labels = torch.ones(len(data_inputs), 1)
data_weights = torch.ones(len(data_inputs), 1)

# SPLIT DATA INTO TRAIN, VAL, TEST SETS
# 70% train, 15% val, 15% test
train_size, val_size, test_size = 0.7, 0.15, 0.15
n_data = len(data_inputs)
indices_data = torch.randperm(n_data)
data_train_idx = indices_data[:int(train_size * n_data)]
data_val_idx = indices_data[int(train_size * n_data):int((train_size + val_size) * n_data)]
data_test_idx = indices_data[int((train_size + val_size) * n_data):]

# SPLIT DY INTO TRAIN, VAL, TEST SETS
n_dy = len(dy_inputs)
indices_dy = torch.randperm(n_dy)
dy_train_idx = indices_dy[:int(train_size * n_dy)]
dy_val_idx = indices_dy[int(train_size * n_dy):int((train_size + val_size) * n_dy)]
dy_test_idx = indices_dy[int((train_size + val_size) * n_dy):]

# OVERSAMPLING DATA TO MATCH DY SIZE 
# only for training set
ratio = int(len(dy_inputs[dy_train_idx]) / len(data_inputs[data_train_idx]))
data_train_inputs = data_inputs[data_train_idx].repeat((ratio, 1))
data_train_labels = data_labels[data_train_idx].repeat((ratio, 1))
data_train_weights = data_weights[data_train_idx].repeat((ratio, 1))

# compare sum of weights
print(f"Sum of DY weights (train): {torch.sum(dy_weights[dy_train_idx]).item():.2f}")
print(f"Sum of Data weights (train, oversampled): {torch.sum(data_train_weights).item():.2f}")

dy_normalization_factor = torch.sum(data_train_weights) / torch.sum(dy_weights[dy_train_idx])
dy_weights[dy_train_idx] *= dy_normalization_factor


# combine data and dy for each set
train_inputs = torch.cat((data_train_inputs, dy_inputs[dy_train_idx]), dim=0)
train_labels = torch.cat((data_train_labels, dy_labels[dy_train_idx]), dim=0)
train_weights = torch.cat((data_train_weights, dy_weights[dy_train_idx]), dim=0)

val_inputs = torch.cat((data_inputs[data_val_idx], dy_inputs[dy_val_idx]), dim=0)
val_labels = torch.cat((data_labels[data_val_idx], dy_labels[dy_val_idx]), dim=0)
val_weights = torch.cat((data_weights[data_val_idx]*ratio, dy_weights[dy_val_idx]), dim=0)   

test_inputs = torch.cat((data_inputs[data_test_idx], dy_inputs[dy_test_idx]), dim=0)
test_labels = torch.cat((data_labels[data_test_idx], dy_labels[dy_test_idx]), dim=0)
test_weights = torch.cat((data_weights[data_test_idx]*ratio, dy_weights[dy_test_idx]), dim=0)

# NORMALIZE INPUTS
# calculate mean and std for each input feature
input_means = torch.mean(train_inputs, dim=0)
input_stds = torch.std(train_inputs, dim=0)
train_inputs = (train_inputs - input_means) / (input_stds + 1e-8) # avoid division by zero
val_inputs = (val_inputs - input_means) / (input_stds + 1e-8)
test_inputs = (test_inputs - input_means) / (input_stds + 1e-8)

# wrap in dataset
train_dataset = TensorDataset(train_inputs, train_labels, train_weights)
val_dataset = TensorDataset(val_inputs, val_labels, val_weights)
test_dataset = TensorDataset(test_inputs, test_labels, test_weights)


# ----------------------------------------------------------------------------------
# SETUP for the NN
class DYClassifierNN(nn.Module):
    def __init__(self):
        super(DYClassifierNN, self).__init__()
        self.fc1 = nn.Linear(5, 50)   # input layer to hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 50)  # hidden layer to hidden layer
        self.fc3 = nn.Linear(50, 1)   # hidden layer to output
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = DYClassifierNN()
lr = 0.0005
lr_threshold = 0.04
loss_fn = nn.BCELoss(reduction='none')
loss_target = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = 4096
n_epochs = 3
dropped_lr = False

# prepare dataloader

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------------------------------------------------------------

plot_score_distribution(model, test_loader, "initial")

# TRAINING LOOP

for epoch in range(n_epochs):
    # --- TRAINING ---
    model.train()
    train_running_loss = 0.0
    # loop over batches
    for batch_idx, (batch_var, batch_labels, batch_weights) in enumerate(train_loader):
        # reset gradients
        optimizer.zero_grad()
        
        
        # get dy weight predictions from the model
        outputs = model(batch_var)

        # loss calculation
        loss = torch.mean(loss_fn(outputs, batch_labels).flatten()*batch_weights.flatten())

        # backpropagation and optimization steps
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"-- Epoch {epoch + 1}/{n_epochs}, "
                  f"Step: {batch_idx + 1}, "
                  f"Loss: {loss.item():.6f}, "
                  f"mean = {outputs.mean().item():.6f}, "
                  f"std = {outputs.std().item():.6f}"
                  )
            if (dropped_lr is False) and (loss.item() <= lr_threshold):
                learn_rate = lr * 0.25  # decrease learning for later epochs
                dropped_lr = True
                print(f"-> Learning rate decreased from {lr} to {learn_rate}!")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learn_rate

            if loss.item() <= loss_target:
                print("-> Early stopping criterion met. Stopping training.")
                break

    
    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_var, val_labels, val_weights in val_loader:
            val_outputs = model(val_var)
            batch_loss = torch.mean(loss_fn(val_outputs, val_labels).flatten()*val_weights.flatten())
            val_loss += batch_loss.item()

    avg_train_loss = train_running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"---- Epoch {epoch + 1}/{n_epochs} : ")
    print(f"Average Training Loss = {avg_train_loss:.6f}")
    print(f"Average Validation Loss = {avg_val_loss:.6f}\n")
    
print("\n---------------- Training completed -----------------")

# ----------------------------------------------------------------------------------

# EVALUATION ON TEST SET

plot_score_distribution(model, test_loader, "final")



