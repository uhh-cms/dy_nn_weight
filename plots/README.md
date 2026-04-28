# Extended Kinematic Distributions

This folder contains the complete set of plots for the Bachelor's Thesis: 
**"Optimising background estimation through Machine Learning techniques for the $HH \rightarrow b\bar{b}\tau^-\tau^+$ analysis with the CMS experiment"**.

 It is structured by the following six Drell-Yan weight sets:

1. **Uncorrected MC**
2. **1D-Fit**
3. **Regression NN**
4. **CARL-TORCH**
5. **CARL-TORCH with DY weights initialised as 1**
6. **all DY weights being equal (normalised to recorded data)**

### Folder Structure
Each weight set folder contains:
- **ee**: Validation channel distributions, including **inclusive plots**.
- **mumu**: Training channel distributions, including **inclusive plots** and folders for **discrete jet/btag multiplicities** (e.g., 2j0b, 3j1b, etc.).

All plots show a comparison between the MC simulation with the corresponding set of DY weights (blue and orange bars) and the recorded data (black dots).
