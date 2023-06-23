#%%TODO: 
import pandas as pd
import matplotlib.pyplot as plt


#%% plot bar graph of results by row
# cols are: run,cindex,pearson,spearman,mse,mae,rmse
df_res = pd.read_csv('results/model_media/DGraphDTA_stats.csv')[2:]
df_res.sort_values(by='run', inplace=True)

df_res.loc[-1] = ['vina', 0.68,0,0,0,0,0]

plt.figure()
bars = plt.bar(df_res['run'],df_res['cindex'])
bars[0].set_color('green')
bars[2].set_color('green')
bars[-1].set_color('red')
plt.title('cindex')
plt.xlabel('run')
plt.xticks(rotation=30)
plt.ylim((0.2, 0.8))
plt.show()
#%%