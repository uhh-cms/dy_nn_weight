import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import order as od
import os
import correctionlib

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
filelist = os.listdir(mypath)

example_file = "w_lnu_1j_pt100to200_amcatnlo.parquet"
fullpath = f"{mypath}{example_file}"
example_array = ak.from_parquet(fullpath)

variables = example_array.fields
print(variables)



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
jet1_pt = config_inst.variables.n.jet1_pt


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
        if var != "event_weight" and not file.startswith("data"):
            values = ak.from_parquet(f"{mypath}{file}")[var]
        else:
            # create event weights = 1 for data
            ll_len = len(ak.from_parquet(f"{mypath}{file}")["ll_pt"])
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


# def create_full_arrays(variable):
#     """
#     Function to read varaible column from all parquet files
#     """

#     data_list = []
#     dy_list = []
#     mc_list = []

#     for file in filelist:
#         # extract the variable values from the file
#         if variable != "event_weight":
#             values = ak.from_parquet(f"{mypath}{file}")[variable]
#         # extract event weights
#         else:
#             if not file.startswith("data_"):
#                 values = ak.from_parquet(f"{mypath}{file}")[variable]
#             else:
#                 # create event weights = 1 for data
#                 ll_len = len(ak.from_parquet(f"{mypath}{file}")["ll_pt"])
#                 values = ak.Array(np.ones(ll_len, dtype=float))

#         # append values to the correct list
#         if file.startswith("data"):
#             data_list.append(values)
#         elif file.startswith("dy"):
#             dy_list.append(values)
#         else:
#             mc_list.append(values)

#     # helper to safely concatenate awkward arrays and return a numpy array
#     def _concat_to_numpy(lst):
#         if not lst:
#             return np.array([], dtype=float)
#         concatenated = ak.concatenate(lst)
#         try:
#             return ak.to_numpy(concatenated)
#         except Exception:
#             return np.asarray(ak.to_list(concatenated), dtype=float)

#     data_arr = _concat_to_numpy(data_list)
#     dy_arr = _concat_to_numpy(dy_list)
#     mc_arr = _concat_to_numpy(mc_list)

#     return data_arr, dy_arr, mc_arr


def filter_events_by_channel(data_arr, dy_arr, mc_arr, desired_channel_id):
    """
    Function to filter the events by channel id and DY region cuts
    -> desired_channel_id: ee = 4, mumu = 5
    """
    data_channel_id, dy_channel_id, mc_channel_id = variable_list["channel_id"]  # noqa: E501
    data_ll_mass, dy_ll_mass, mc_ll_mass = variable_list["ll_mass"]  # noqa: E501
    data_met, dy_met, mc_met = variable_list["met"]

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
    data_arr, dy_arr, mc_arr = variable_list[var_name]
    data_weights, dy_weights_original, mc_weights = variable_list["event_weight"]  # noqa: E501

    # use original DY weights if none are provided
    if dy_weights is None:
        print("--> Using original DY event weights!")
        dy_weights = dy_weights_original
    else:
        print("--> Using updated DY event weights!")
        dy_weights = dy_weights

    # filter arrays considering ee channel id and DY region cuts
    if filter_events:
        data_arr, dy_arr, mc_arr = filter_events_by_channel(data_arr, dy_arr, mc_arr, 4)  # noqa: E501
        data_weights, dy_weights, mc_weights = filter_events_by_channel(  # noqa: E501
            data_weights, dy_weights, mc_weights, 4
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
    print(f"Plot saved as plot_{file_version}.pdf")

# --------------------------------------------------------------------------------------------------
# MAIN SCRIPT


# get DY weights from json
_, dy_n_jet, _ = variable_list["n_jet"]
_, dy_n_tag, _ = variable_list["n_btag_pnet"]
_, dy_event_weight, _ = variable_list["event_weight"]
_, dy_ll_pt, _ = variable_list["ll_pt"]
weight = correction_set.evaluate(era, dy_n_jet, dy_n_tag, dy_ll_pt, syst)
correctionlib_weight = weight * dy_event_weight

# use original DY weights
plot_function(jet1_pt, "original_dy_weights_jet1_pt")

plot_function(bb_pt, "original_dy_weights_bb_pt")   

# use updated DY weights form json file
plot_function(jet1_pt, "correctionlib_weights_jet1_pt", dy_weights=correctionlib_weight)

plot_function(bb_pt, "correctionlib_weights_bb_pt", dy_weights=correctionlib_weight)




