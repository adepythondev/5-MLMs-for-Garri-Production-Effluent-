import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("/storage/emulated/0/FUFU PRODUCTION EFFLUENT.xlsx", engine='openpyxl')
df = df.drop(columns=['Unnamed: 12'], errors='ignore')
df = df.reset_index(drop=True)
df['TIME_INDEX'] = df.index

known_v = df['VOLTAGE (V)'].notna().sum()
known_c = df['CURRENT (I) Ma'].notna().sum()
known_ph = df['pH'].notna().sum()
known_t = df['ROOM TEMPERATURE (°C) '].notna().sum()

scaler = StandardScaler()

if df['VOLTAGE (V)'].isna().sum() > 0:
    known = df[df['VOLTAGE (V)'].notna()]
    missing = df[df['VOLTAGE (V)'].isna()]
    X_train = scaler.fit_transform(known[['TIME_INDEX']])
    X_pred = scaler.transform(missing[['TIME_INDEX']])
    m = SVR(kernel='rbf', C=100, gamma=0.1)
    m.fit(X_train, known['VOLTAGE (V)'])
    df.loc[df['VOLTAGE (V)'].isna(), 'VOLTAGE (V)'] = np.round(m.predict(X_pred), 3)

if df['CURRENT (I) Ma'].isna().sum() > 0:
    known = df[df['CURRENT (I) Ma'].notna()]
    missing = df[df['CURRENT (I) Ma'].isna()]
    X_train = scaler.fit_transform(known[['TIME_INDEX']])
    X_pred = scaler.transform(missing[['TIME_INDEX']])
    m = SVR(kernel='rbf', C=100, gamma=0.1)
    m.fit(X_train, known['CURRENT (I) Ma'])
    df.loc[df['CURRENT (I) Ma'].isna(), 'CURRENT (I) Ma'] = np.round(m.predict(X_pred), 3)

if df['ROOM TEMPERATURE (°C) '].isna().sum() > 0:
    known = df[df['ROOM TEMPERATURE (°C) '].notna()]
    missing = df[df['ROOM TEMPERATURE (°C) '].isna()]
    X_train = scaler.fit_transform(known[['TIME_INDEX']])
    X_pred = scaler.transform(missing[['TIME_INDEX']])
    m = SVR(kernel='rbf', C=100, gamma=0.1)
    m.fit(X_train, known['ROOM TEMPERATURE (°C) '])
    df.loc[df['ROOM TEMPERATURE (°C) '].isna(), 'ROOM TEMPERATURE (°C) '] = np.round(m.predict(X_pred), 3)

if df['pH'].isna().sum() > 0:
    known = df[df['pH'].notna()]
    missing = df[df['pH'].isna()]
    X_train = scaler.fit_transform(known[['TIME_INDEX', 'VOLTAGE (V)', 'CURRENT (I) Ma']])
    X_pred = scaler.transform(missing[['TIME_INDEX', 'VOLTAGE (V)', 'CURRENT (I) Ma']])
    m = SVR(kernel='rbf', C=100, gamma=0.1)
    m.fit(X_train, known['pH'])
    df.loc[df['pH'].isna(), 'pH'] = np.round(m.predict(X_pred), 3)

df = df.drop(columns=['TIME_INDEX'])
df.to_excel("/storage/emulated/0/FUFU_SVR.xlsx", engine='openpyxl', index=False)

fig, axes = plt.subplots(4, 1, figsize=(12, 18))
fig.suptitle('Fufu Effluent - Support Vector Regression', fontsize=13, fontweight='bold')
axes[0].plot(df.index[:known_v], df['VOLTAGE (V)'][:known_v], 'g-o', markersize=4, label='Known')
axes[0].plot(df.index[known_v:], df['VOLTAGE (V)'][known_v:], 'g--o', markersize=4, label='Predicted')
axes[0].set_title('Voltage'); axes[0].set_ylabel('Voltage (V)'); axes[0].legend(); axes[0].grid(True)
axes[1].plot(df.index[:known_c], df['CURRENT (I) Ma'][:known_c], 'b-o', markersize=4, label='Known')
axes[1].plot(df.index[known_c:], df['CURRENT (I) Ma'][known_c:], 'b--o', markersize=4, label='Predicted')
axes[1].set_title('Current'); axes[1].set_ylabel('Current (mA)'); axes[1].legend(); axes[1].grid(True)
axes[2].plot(df.index[:known_ph], df['pH'][:known_ph], 'r-o', markersize=4, label='Known')
axes[2].plot(df.index[known_ph:], df['pH'][known_ph:], 'r--o', markersize=4, label='Predicted')
axes[2].set_title('pH'); axes[2].set_ylabel('pH'); axes[2].legend(); axes[2].grid(True)
axes[3].plot(df.index[:known_t], df['ROOM TEMPERATURE (°C) '][:known_t], 'm-o', markersize=4, label='Known')
axes[3].plot(df.index[known_t:], df['ROOM TEMPERATURE (°C) '][known_t:], 'm--o', markersize=4, label='Predicted')
axes[3].set_title('Temperature'); axes[3].set_ylabel('Temp (°C)'); axes[3].set_xlabel('Time Step'); axes[3].legend(); axes[3].grid(True)
plt.tight_layout()
plt.savefig('/storage/emulated/0/FUFU_SVR_plots.png', dpi=150, bbox_inches='tight')
print("SVR done! Check FUFU_SVR.xlsx and FUFU_SVR_plots.png")