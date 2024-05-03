# %%
from src.analysis.figures import tbl_dpkd_metrics_overlap, tbl_dpkd_metrics_n_mut, tbl_dpkd_metrics_in_binding, predictive_performance
from src.analysis.metrics import get_metrics

_ = predictive_performance(compare_overlap=True, verbose=True, plot=True, NORMALIZE=False)

# %%
_ = predictive_performance(compare_overlap=True, verbose=True, plot=True, NORMALIZE=True)

# %%
tbl_dpkd_metrics_overlap()

#%%
tbl_dpkd_metrics_n_mut()

#%%
tbl_dpkd_metrics_in_binding()


#%