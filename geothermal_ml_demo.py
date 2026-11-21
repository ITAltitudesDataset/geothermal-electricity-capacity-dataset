"""Educational Isolation Forest demo for a 50-row geothermal capacity preview."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

HERE = Path(__file__).resolve().parent
DATA = HERE / 'Geothermal_50_Row_Buyer_Preview.csv'
df = pd.read_csv(DATA)
assert len(df) == 50 and {'Country','ResourceType','GridConnection','Year','CapacityMW'}.issubset(df.columns)
usable = df[df['CapacityMW'].notna()].copy()
# log1p handles strongly skewed nonnegative capacity measurements.
assert (usable['CapacityMW'] >= 0).all()
X = np.log1p(usable[['CapacityMW']].astype(float))
model = IsolationForest(n_estimators=200, contamination=0.10, random_state=42)
usable['AnomalyFlag'] = np.where(model.fit_predict(X) == -1, 'Flagged', 'Not flagged')
usable['AnomalyScore'] = model.decision_function(X).round(5)
result = usable[['Country','Year','GridConnection','CapacityMW','AnomalyFlag','AnomalyScore']].sort_values('CapacityMW',ascending=False)
result.to_csv(HERE/'geothermal_ml_results.csv',index=False)
fig,ax=plt.subplots(figsize=(10,5.6))
colors=np.where(usable['AnomalyFlag'].eq('Flagged'),'#d97732','#2563a6')
ax.scatter(usable['Year'],usable['CapacityMW'],c=colors,s=75,alpha=.85,edgecolor='white',linewidth=.5)
ax.set_yscale('symlog',linthresh=1)
ax.set_title('Geothermal capacity: exploratory anomaly detection (50-row preview)',fontsize=13,weight='bold')
ax.set_xlabel('Reported year');ax.set_ylabel('Capacity (MW; symlog scale)')
ax.grid(alpha=.18)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#2563a6',markersize=9,label='Not flagged'),Line2D([0],[0],marker='o',color='w',markerfacecolor='#d97732',markersize=9,label='Flagged')])
fig.text(.12,.015,'Model: Isolation Forest on log(1 + CapacityMW). Flags are statistical, not verified errors.',fontsize=8)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(HERE/'geothermal_ml_chart.png',dpi=180);plt.close(fig)
print(f'Input rows: {len(df)} | Numeric capacity rows: {len(usable)} | Missing capacity: {df.CapacityMW.isna().sum()} | Flagged: {usable.AnomalyFlag.eq("Flagged").sum()}')
print(result[result.AnomalyFlag.eq('Flagged')][['Country','Year','CapacityMW']].to_string(index=False))
