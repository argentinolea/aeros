import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import seaborn as sns


# Carica dati reali e simulati
input_file_path = "../co2_false_negative_sim/65m3_export_false_negative.csv"
data_df_real = pd.read_csv(input_file_path, delimiter=",")
merged_df = data_df_real[["temperature", "co2", "humidity", "volume", "ventilation rate", "occupants_modified"]].round(2)

##############################
# 2. PRETRAINING (SU DATI SIMULATI)
##############################

# Calcolo della varianza (filtraggio per varianza CO2 tra 0 e 20)

merged_df = merged_df[[
    "temperature", "co2", "humidity",
    "volume", "ventilation rate", "occupants_modified"
]].round(2)

merged_df = merged_df[
    (merged_df["temperature"] > 20) & (merged_df["temperature"] < 25) &
    (merged_df["humidity"] > 20) & (merged_df["humidity"] < 80) &
    (merged_df["volume"] > 55) & (merged_df["volume"] < 75)
]

# Aggiungiamo la feature CO2_variance e definiamo le feature da usare
feature_columns = ["temperature", "humidity", "ventilation rate", "volume","co2"]
target_col = ["occupants_modified"]
X = merged_df[feature_columns]
y = merged_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

if r2 < 0:
    negative_r2_clusters.append(cluster_id)

print(f"   MAE  (Mean Absolute Error)      : {mae:.2f} ppm")
print(f"   MSE  (Mean Squared Error)       : {mse:.2f} ppm²")
print(f"   RMSE (Root Mean Squared Error)  : {rmse:.2f} ppm")
print(f"   R²   (Coefficient of Determination): {r2:.4f}")

sensor_data_1 = pd.DataFrame([
    {"temperature": 22.35, "humidity": 43.5, "ventilation rate": 0.25, "volume": 65.0, "co2": 1150.0}
])

predicted_occupancy = model.predict(sensor_data_1)
print(f"Predicted number of occupants: {predicted_occupancy}")


plot = sns.relplot(x='occupants_modified', y='co2', data=merged_df, 
            height=3.8, aspect=1.8, kind='scatter')
sns.set_style('dark')
plot.fig.savefig("occupancy_vs_co2.png", dpi=300)