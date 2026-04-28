from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import awkward as ak
import order as od
import os
import torch
from matplotlib.ticker import LogLocator
from matplotlib.patches import Patch
from modules.hbt.hbt.config.analysis_hbt import analysis_hbt
from modules.hbt.modules.columnflow.columnflow.hist_util import create_hist_from_variables, fill_hist

# --- SETUP ---
year = "22pre_v14"
config_inst = analysis_hbt.get_config(year)
mypath = f"/data/dust/user/riegerma/hh2bbtautau/dy_dnn_data/inputs_prod20_vbf/{year}/"  # noqa: E501
filelist = os.listdir(mypath)
example_file = "w_lnu_1j_pt100to200_amcatnlo.parquet"
fullpath = f"{mypath}{example_file}"
example_array = ak.from_parquet(fullpath)
variables = example_array.fields
era = year.replace("22", "2022").replace("_v14", "EE")

# Prepare variables
ll_mass = config_inst.variables.n.dilep_mass
ll_pt = config_inst.variables.n.dilep_pt

bb_mass = config_inst.variables.n.dibjet_mass
bb_pt = config_inst.variables.n.dibjet_pt

n_jet = config_inst.variables.n.njets
n_btag_pnet = config_inst.variables.n.nbjets_pnet

met_pt = config_inst.variables.n.met_pt

jet1_pt = config_inst.variables.n.jet1_pt

e1_pt = config_inst.variables.n.e1_pt
mu1_pt = config_inst.variables.n.mu1_pt

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

def _concat_to_numpy(lst):
    if not lst:
        return np.array([], dtype=float)
    concatenated = ak.concatenate(lst)
    try:
        return ak.to_numpy(concatenated)
    except Exception:
        return np.asarray(ak.to_list(concatenated), dtype=float)

variable_list = {}
for var in variables:
    data_arr = _concat_to_numpy(temp_storage["data"][var])
    dy_arr = _concat_to_numpy(temp_storage["dy"][var])
    mc_arr = _concat_to_numpy(temp_storage["mc"][var])

    variable_list[var] = [data_arr, dy_arr, mc_arr]

# --- FUNCTIONS ---

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

def calculate_reduced_chi2(data_hist, total_mc_hist):
    observed = data_hist.values()
    expected = total_mc_hist.values()
    total_variance = data_hist.variances() + total_mc_hist.variances()
    bin_weights = data_hist.values() / np.sum(data_hist.values())
    mask = total_variance > 0
    
    #chi2 = np.sum(((observed[mask] - expected[mask])**2) / total_variance[mask] * bin_weights[mask]) # weighted by bin
    chi2 = np.sum(((observed[mask] - expected[mask])**2) / total_variance[mask]) # normal reduced chi2 without bin weighting
    return chi2

# --- ANALYSIS LOOP ---

test_vars = [ll_pt, bb_pt, bb_mass, n_jet, n_btag_pnet, ll_mass, met_pt]
var_names = [r"$p_{T}(\ell\ell)$", r"$p_{T}(bb)$", r"$m(bb)$", r"$N_{jets}$", r"$N_{b-tags}$", r"$m(\ell\ell)$", r"$p_{T}^{miss}$"]
variable_markers = ["s", "^", "v", "P", "X", "D", "o"]

method_colors = {
    "Uncorrected MC": "firebrick",
    "1D-Fit": "tomato",
    "Regression NN": "deepskyblue",
    "CARL Torch": "blue"
}

# --- LOAD DATA ---

# Load the saved weights
w_regression = np.load("dy_event_weights_regression.npy")
w_correctionlib = np.load("dy_event_weights_correctionlib.npy")
w_carl_mumu = np.load("dy_event_weights_carl.npy")
w_carl_ee = np.load("dy_event_weights_carl_ee.npy")

methods_weights_mumu = {
    "Uncorrected MC": variable_list["event_weight"][1], 
    "1D-Fit": w_correctionlib,
    "Regression NN": w_regression,
    "CARL Torch": w_carl_mumu,
}

methods_weights_ee = {
    "Uncorrected MC": variable_list["event_weight"][1], 
    "1D-Fit": w_correctionlib,
    "Regression NN": w_regression,
    "CARL Torch": w_carl_ee,
}


method_names = list(methods_weights_mumu.keys())

print("Calculating Reduced Chi-Square for all methods...")

chi2_results_mumu = {var.name: [] for var in test_vars}
chi2_results_ee = {var.name: [] for var in test_vars}

for var in test_vars:
    var_name = var.name.replace("dilep", "ll").replace("dibjet", "bb").replace("njets", "n_jet").replace("nbjets", "n_btag")
    
    # Calculate chi2 for inclusive selection (no n_jet cut)
    # MUMU
    data_arr_5, dy_arr_5, mc_arr_5, _, dy_mask_5, _ = filter_events_by_channel(*variable_list[var_name], 5, return_masks=True)
    data_weights_5, _, mc_weights_5 = filter_events_by_channel(*variable_list["event_weight"], 5)
    hist_data_5 = hist_function(var, data_arr_5, data_weights_5)
    hist_mc_5  = hist_function(var, mc_arr_5, mc_weights_5)
    
    # EE
    data_arr_4, dy_arr_4, mc_arr_4, _, dy_mask_4, _ = filter_events_by_channel(*variable_list[var_name], 4, return_masks=True)
    data_weights_4, _, mc_weights_4 = filter_events_by_channel(*variable_list["event_weight"], 4)
    hist_data_4 = hist_function(var, data_arr_4, data_weights_4)
    hist_mc_4  = hist_function(var, mc_arr_4, mc_weights_4)


    for m_name in method_names:
        # MuMu 
        dy_weights_5_filtered = methods_weights_mumu[m_name][dy_mask_5]
        total_mc_5 = hist_function(var, dy_arr_5, dy_weights_5_filtered) + hist_mc_5
        chi2_results_mumu[var.name].append(calculate_reduced_chi2(hist_data_5, total_mc_5))

        # EE
        dy_weights_4_filtered = methods_weights_ee[m_name][dy_mask_4]
        total_mc_4 = hist_function(var, dy_arr_4, dy_weights_4_filtered) + hist_mc_4
        chi2_results_ee[var.name].append(calculate_reduced_chi2(hist_data_4, total_mc_4))

    # Calculate chi2 for n_jet 2 and 3
    for nj in [2, 3]:

        # MUMU
        data_arr_5, dy_arr_5, mc_arr_5, _, dy_mask_5, _ = filter_events_by_channel(*variable_list[var_name], 5, return_masks=True, n_jet_cut=nj)
        data_weights_5, _, mc_weights_5 = filter_events_by_channel(*variable_list["event_weight"], 5, n_jet_cut=nj)
        hist_data_5 = hist_function(var, data_arr_5, data_weights_5)
        hist_mc_5  = hist_function(var, mc_arr_5, mc_weights_5)
        
        # EE
        data_arr_4, dy_arr_4, mc_arr_4, _, dy_mask_4, _ = filter_events_by_channel(*variable_list[var_name], 4, return_masks=True, n_jet_cut=nj)
        data_weights_4, _, mc_weights_4 = filter_events_by_channel(*variable_list["event_weight"], 4, n_jet_cut=nj)
        hist_data_4 = hist_function(var, data_arr_4, data_weights_4)
        hist_mc_4  = hist_function(var, mc_arr_4, mc_weights_4)

        chi2_results_nj_ee = []
        chi2_results_nj_mumu = []
        for m_name in method_names:
            # MuMu 
            dy_weights_5_filtered = methods_weights_mumu[m_name][dy_mask_5]
            total_mc_5 = hist_function(var, dy_arr_5, dy_weights_5_filtered) + hist_mc_5
            chi2_results_nj_mumu.append(calculate_reduced_chi2(hist_data_5, total_mc_5))

            # EE
            dy_weights_4_filtered = methods_weights_ee[m_name][dy_mask_4]
            total_mc_4 = hist_function(var, dy_arr_4, dy_weights_4_filtered) + hist_mc_4
            chi2_results_nj_ee.append(calculate_reduced_chi2(hist_data_4, total_mc_4))

        chi2_results_mumu[f"{var.name}_nj{nj}"] = chi2_results_nj_mumu
        chi2_results_ee[f"{var.name}_nj{nj}"] = chi2_results_nj_ee

def create_chi2_plot(chi2_dict, channel_name, filename):
    print(f"--- Plotting {channel_name} categorical results ---")
    plt.figure(figsize=(14, 8))
    
    vars_to_plot = [v.name for v in test_vars]
    x_positions = np.arange(len(vars_to_plot))

    # Mapping for clustering
    # inclusive = 0 offset, nj2 = +0.15 offset, nj3 = +0.3 offset
    regions = {
        "inclusive": {"suffix": "", "offset": 0.0, "hatch": None, "fill": True},
        "nj2": {"suffix": "_nj2", "offset": 0.2, "hatch": "////", "fill": False},
        "nj3": {"suffix": "_nj3", "offset": 0.4, "hatch": None, "fill": False}
    }
    
    method_handles = []

    ideal_line = plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, zorder=0)

    # Loop over methods
    for m_idx, m_name in enumerate(method_names):
        m_color = method_colors[m_name]
        
        # Loop over variables for this method
        for v_idx, var_name in enumerate(vars_to_plot):
            marker_shape = variable_markers[v_idx]
            
            for region_name, region_info in regions.items():
                # Construct the variable name with region suffix
                region_var_name = f"{var_name}{region_info['suffix']}"
                
                if region_var_name not in chi2_dict:
                    continue  # Skip if this region is not available for this variable

                chi2_val = chi2_dict[region_var_name][m_idx]

                fcolor = m_color if region_info["fill"] else 'none'
                p = plt.scatter(
                    x_positions[v_idx] + region_info["offset"], 
                    chi2_val, 
                    facecolor=fcolor,
                    edgecolor=m_color, 
                    marker=marker_shape, 
                    s=120,            # Marker size
                    linewidths=1.2,
                    hatch=region_info["hatch"],
                    alpha=0.8,
                    zorder=m_idx+1  # Ensure methods are layered in a consistent order
                )
            
                if v_idx == 0 and region_name == "inclusive":  # Only add to method legend once
                    method_handles.append(p)

    # ---Formatting---
    plt.yscale("symlog", linthresh=1.0, linscale=0.2)
    #plt.ylim(-1, 2500) # for chi2 weighted by bin
    plt.ylim(-1, 4000) # for normal chi2 without bin weighting
    plt.xticks(x_positions+0.15, var_names, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Kinematic Variables", fontsize=18, loc='right')
    plt.ylabel(r"Reduced $\chi^{2}$", fontsize=18, loc='top')
    plt.title(f"Model Performance by Kinematic Variable - {channel_name} Channel", fontsize=20)

    # ---Method Legend---
    leg1_names = method_names + ["Ideal Modeling (1.0)"]
    leg1_handles = method_handles + [ideal_line]
    leg1 = plt.legend(leg1_handles, leg1_names, loc='lower right', fontsize=14, framealpha=0.9)
    plt.gca().add_artist(leg1)

    # ---Jet Region Legend---
    jet_proxies = [
        Patch(facecolor='grey', edgecolor='black', alpha=0.8, label='Inclusive'),
        Patch(facecolor='none', edgecolor='black', alpha=0.8, hatch='////', label=r'$N_{jets}=2$'),
        Patch(facecolor='none', edgecolor='grey', alpha=0.8, label=r'$N_{jets}=3$')
    ]
    plt.legend(handles=jet_proxies, loc='lower left', title="Jet Multiplicity Cuts", fontsize=14, framealpha=0.9)

    # ---Gridlines---
    plt.grid(True, which="both", axis='y', color='grey', linestyle='-', alpha=0.3, linewidth=0.7)
    plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12))
    # We hide the minor ticks below 1.0 manually
    plt.gcf().canvas.draw() 
    ticks = plt.gca().yaxis.get_minor_ticks()
    for tick in ticks:
        if tick.get_loc() < 1.0:
            tick.gridline.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

create_chi2_plot(chi2_results_mumu, r"$\mu\mu$", "chi2_mumu.png")
create_chi2_plot(chi2_results_ee, r"$ee$", "chi2_ee.png")

# --- PRINT STATISTICS ---
orig_idx = 0

input_variables = ["dilep_pt", "dibjet_pt", "dibjet_mass", "njets", "nbjets_pnet"]

def print_stats(chi2_dict, channel_name):
    print(f"\n{'='*80}")
    print(f"STATISTICS FOR {channel_name.upper()} (Input Variables Only)")
    print(f"{'='*80}")
    
    # Table Header
    print(f"{'Variable':<15} | {'Method':<15} | {'Chi2':<10} | {'% Dev from Orig'}")
    print("-" * 65)
    
    # Store means for the specific input variables
    method_means = {m: [] for m in method_names}
    
    # Sort variables to match your test_vars order if possible
    for var_name in chi2_dict:
        # SKIP variables that are not in your 5 input variables
        if var_name not in input_variables:
            continue
            
        var_values = chi2_dict[var_name]
        orig_val = var_values[orig_idx]
        
        for i, m_name in enumerate(method_names):
            current_val = var_values[i]
            # Calculate % deviation: ((New - Old) / Old) * 100
            percent_dev = ((current_val - orig_val) / orig_val) * 100
            
            method_means[m_name].append(current_val)
            
            print(f"{var_name:<15} | {m_name:<15} | {current_val:<10.3f} | {percent_dev:>+8.2f}%")
        print("-" * 65)

# Execute for both channels
print_stats(chi2_results_mumu, "MuMu")
print_stats(chi2_results_ee, "EE")