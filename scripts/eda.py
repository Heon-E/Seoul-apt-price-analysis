import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data/merged.csv')
df['date'] = pd.to_datetime(df['date'])

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df['date'], df['base_rate'], color='blue', label='기준금리')
ax2 = ax1.twinx()
ax2.plot(df['date'], df['seoul_apt_price_index'], color='red', label='아파트 가격지수')
ax1.set_ylabel('기준금리 (%)', color='blue')
ax2.set_ylabel('아파트 가격지수', color='red')
plt.title('기준금리 vs 서울 아파트 가격지수 (2015-2025)')
fig.legend(loc='upper left')
plt.savefig('figures/timeseries.png', dpi=150, bbox_inches='tight')
plt.show()
print(df[['base_rate', 'seoul_apt_price_index']].corr())
import statsmodels.api as sm
X = df['base_rate']
y = df['mom_change_pct']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
fig2, ax3 = plt.subplots(figsize=(8, 5))
ax3.scatter(df['base_rate'], df['mom_change_pct'], color='gray', alpha=0.5)
ax3.plot(df['base_rate'], model.fittedvalues, color='red', linewidth=2)
ax3.set_xlabel('기준금리 (%)')
ax3.set_ylabel('집값 월간 상승률 (%)')
plt.title('기준금리 vs 집값 월간 상승률')
plt.savefig('figures/scatter.png', dpi=150, bbox_inches='tight')
plt.show()