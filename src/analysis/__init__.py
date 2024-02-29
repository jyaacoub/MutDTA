from src.analysis.metrics import get_metrics
from src.analysis.utils import count_missing_res


#%%
if __name__ == '__main__':
    #%% visualizing and analyzing docking results
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr
    import pandas as pd
    import numpy as np
    import random, os
    from src.analysis.metrics import get_metrics


    # cindex:
    if os.path.basename(os.getcwd()) == 'analysis':

        import os; os.chdir('../../') # for if running from src/analysis/
    print(os.getcwd())

    # CASF-2012 core dataset
    # core_2012_filter = []
    # with open('data/PDBbind/2012_core_data.lst', 'r') as f:
    #     for line in f.readlines():
    #         if '#' == line[0]: continue
    #         code = line[:4]
    #         core_2012_filter.append(code)
    # core_2012_filter = pd.DataFrame(core_2012_filter, columns=['PDBCode'])

    # filter = pd.read_csv('results/PDBbind/vina_out/run9.csv')['PDBCode'] #core_2012_filter['PDBCode']

    # Filter out for test split only
    np.random.seed(0)
    random.seed(0)
    df_x = pd.read_csv('data/PDBbind/kd_ki/X.csv') 
    pdbcodes = np.array(df_x['PDBCode'])
    random.shuffle(pdbcodes)
    _, filter = np.split(df_x['PDBCode'], [int(.8*len(df_x))])

    # %%
    save = True
    for run_num in [10]:
        print(f'run{run_num}:')
        y_path = 'data/PDBbind/kd_ki/Y.csv'
        vina_out = f'results/PDBbind/vina_out/run{run_num}.csv'
        save_path = 'results/PDBbind/media/kd_ki'

        ##%%
        vina_pred = pd.read_csv(vina_out)
        actual = pd.read_csv(y_path)
        # vina_pred = vina_pred.merge(filter, on='PDBCode') # filter out for test split only
        
        mrgd = actual.merge(vina_pred, on='PDBCode')
        y = mrgd['affinity'].values
        z = mrgd['vina_kd(uM)'].values
        log_y = -np.log(y*1e-6)
        log_z = -np.log(z*1e-6)

        get_metrics(log_y, log_z, 
                    save_figs=save,
                    save_path=save_path,
                    model_key=f'{run_num}',
                    csv_file=f'{save_path}/vina_stats.csv',
                    
                    show=True,
                    title_prefix=f'run{run_num} - ',)
        
    
#%% plot bar graph of results by row (comparing vina to DGraphDTA)
# cols are: run,cindex,pearson,spearman,mse,mae,rmse
# df_res = pd.read_csv('results/model_media/DGraphDTA_stats.csv')[2:]
# df_res.sort_values(by='run', inplace=True)

# df_res.loc[-1] = ['vina', 0.68,0.508,0.520,17.812,3.427,4.220] # hard coded vina results

# for col in df_res.columns[1:]:
#     plt.figure()
#     bars = plt.bar(df_res['run'],df_res[col])
#     bars[0].set_color('green')
#     bars[2].set_color('green')
#     bars[-1].set_color('red')
#     plt.title(col)
#     plt.xlabel('run')
#     plt.xticks(rotation=30)
#     # plt.ylim((0.2, 0.8))
#     plt.show()