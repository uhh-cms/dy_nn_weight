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
# SETUP

# choose which year to use
year = "22pre_v14"

# define file path
config_inst = analysis_hbt.get_config(year)
mypath = f"/data/dust/user/riegerma/hh2bbtautau/dy_dnn_data/inputs_prod20_vbf/{year}/"  # noqa: E501
example_file = "w_lnu_1j_pt100to200_amcatnlo.parquet"
fullpath = f"{mypath}{example_file}"
example_array = ak.from_parquet(fullpath)

# load DY weight corrections from json file
dy_file = "/afs/desy.de/user/a/alvesand/public/dy_corrections.json.gz"
dy_correction = correctionlib.CorrectionSet.from_file(dy_file)
correction_set = dy_correction["dy_weight"]
syst = "nom"
era = year.replace("22", "2022").replace("_v14", "EE")

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

met_pt = config_inst.variables.n.met_pt
met_phi = config_inst.variables.n.met_phi


# get processes
bkg_process = od.Process(name="mc", id="+", color1="#e76300", label="MC")
dy_process = config_inst.get_process("dy")
data_process = config_inst.get_process("data")

# --------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS


def create_full_arrays(variable):
    """
    Function to read varaible column from all parquet files
    """
    filelist = os.listdir(mypath)

    data_list = []
    dy_list = []
    mc_list = []

    for file in filelist:
        # extract the variable values from the file
        if variable != "event_weight":
            values = ak.from_parquet(f"{mypath}{file}")[variable]
        # extract event weights
        else:
            if not file.startswith("data_"):
                values = ak.from_parquet(f"{mypath}{file}")[variable]
            else:
                # create event weights = 1 for data
                ll_len = len(ak.from_parquet(f"{mypath}{file}")["ll_pt"])
                values = ak.Array(np.ones(ll_len, dtype=float))

        # append values to the correct list
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


def filter_events_by_channel(data_arr, dy_arr, mc_arr, desired_channel_id):
    """
    Function to filter the events by channel id and DY region cuts
    -> desired_channel_id: ee = 4, mumu = 5
    """
    data_channel_id, dy_channel_id, mc_channel_id = create_full_arrays("channel_id")  # noqa: E501
    data_ll_mass, dy_ll_mass, mc_ll_mass = create_full_arrays("ll_mass")  # noqa: E501
    data_met, dy_met, mc_met = create_full_arrays("met_pt")

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

    return data_arr[data_mask], dy_arr[dy_mask], mc_arr[mc_mask]


def hist_function(var_instance, data, weights):
    h = create_hist_from_variables(var_instance, weight=True)
    fill_hist(h, {var_instance.name: data, "weight": weights})
    return h


def plot_function(var_instance: od.Variable, file_version: str, dy_weights=None, filter_events: bool = True):  # noqa: E501
    var_name = var_instance.name.replace("dilep", "ll").replace("dibjet", "bb")

    # get variable arrays and original event weights from files
    data_arr, dy_arr, mc_arr = create_full_arrays(var_name)
    data_weights, dy_weights_original, mc_weights = create_full_arrays("event_weight")  # noqa: E501

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

# --------------------------------------------------------------------------------------------------
# MAIN SCRIPT


# define bins and bin function
bin_tupples = [(), (), ...]
def bin_data(array_to_bin, weight_to_bin, bins=bin_tupples):
    # function should turn the arrays [...] into a list of arrays [[...] , [...], ...] of len(list)) = len(bins)
    # where the first array in the list only contains the events that should fall in the first pt_ll bin, etc.

    # binned_arrays = ....
    # binned_weights = ....

    return binned_arrays, binned_weights


# read pt_ll and event weights from files

# bin the pt_ll and event weights arrays using the bin function

# calculate number of expected DY events in each bin by doing Data-MC

# calculate number of actual DY events in each bin by doing sum(event_weights)_bin

# convert numpy/awkard arrays into torch tensors

# define class model
# use a simple feedforward NN with 1-10-1 architecture, ReLu activation

# setup the learning rate, loss function (use MSE loss) and optimizer
# ....

# training loop for 15 epochs at first
for epoch in range(15):

    # set gradients to zero

    # define loss = loss_bin1 + loss_bin2 + ...
    # where loss_bin1 should compare the number of DY events in bin1 after reweighting to the original number of DY events

    # do backpropagation and optimization step

    print(f"epoch {epoch}: Loss = {loss.item()}")
