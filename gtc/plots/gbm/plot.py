import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import os

# to be used with notebook
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
sns.set_style("whitegrid")

df = pd.read_csv('results.txt', delimiter=r'\s+', comment='#', quotechar='"', error_bad_lines = False)

# coerces TBD to Nan and treated as number
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
df['AUC'] = pd.to_numeric(df['AUC'], errors='coerce')

# dropping TBD values, can be added when info is available
df.dropna(inplace=True)
df.set_index('hardware', inplace = True)

for task in ["GBM"]:
  mydf = df.loc[df["task"]==task]
  mydf["gpu"] = mydf.index.str.contains("GPU")
  print(mydf)

  #colors = {'gpu':'green', 'cpu':'blue'}
  colors = {'gpu':'limegreen', 'cpu':'dodgerblue'}

  plt.figure()
  ax = mydf['runtime'].plot.barh(
              figsize = (16,9), 
              width = 0.7,  ## bars
              logx = False, color = list(mydf["gpu"].apply(lambda x:colors['gpu' if x else 'cpu']))) 
  ax.set_xlim([0,4800])

  plt.title('H2O.ai Machine Learning $-$ Gradient Boosting Machine', fontsize = 34, y=1.1)
  plt.annotate('Time to Train 16 H2O XGBoost Models (histogram method)', (0,0), (470,-0.8), fontsize = 24)
  ax.tick_params(labelsize = 24)
  ax.set_xlabel('time [sec]', fontsize = 24, labelpad=15, fontweight = 'bold')
  ax.grid(linewidth=0)
  ax.set_ylabel('hardware', fontsize = 0)

  # # axis can be inverted using below line
  ax.invert_yaxis()
  plt.tight_layout()

  plt.annotate("http://github.com/h2oai/perf/", (0,0), (-170, -98), fontsize = 16, xycoords='axes fraction', textcoords='offset points', va='top', ha='left')

  footnote_text=\
"*NVIDIA DGX-1, **Dual Intel Xeon E5-2698 v4"\
"\nHiggs dataset (binary classification): 1M rows, 29 cols; max_depth: {6,8,10,12}, sample_rate: {0.7,0.8,0.9,1.0}"
  plt.annotate(footnote_text, (0,0), (910, -80), fontsize = 16, xycoords='axes fraction', textcoords='offset points', va='top', ha='right')

  for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[:,1]
    ax.annotate('{:.0f}'.format(x[1]), (x[1]+30, y.mean()), ha='left', va='center', fontsize = 20, color = 'k', fontweight = 'bold')

  green_patch = mpatches.Patch(color='limegreen', label='GPU')
  blue_patch = mpatches.Patch(color='dodgerblue', label='CPU')
  ax.legend(handles = [green_patch, blue_patch], markerscale = 10, fontsize = 20, loc = 4, frameon=True)
  plt.gcf().subplots_adjust(bottom=0.2)
#  green_patch = mpatches.Patch(color='limegreen', label='GPU')
#  blue_patch = mpatches.Patch(color='dodgerblue', label='CPU')
#  ax.legend(handles = [green_patch, blue_patch], markerscale = 10, fontsize = 20, loc = 4)

  plt.savefig("results" + task + "-tmp.png", dpi=300)
  os.system("convert resultsGBM-tmp.png logo.png -composite resultsGBM.png")
  os.system("rm resultsGBM-tmp.png")
  plt.close()
