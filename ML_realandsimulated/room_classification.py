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

def lr_metrics(merged_df):
    features = [ "temperature", "humidity", "volume", "ventilation rate"]
    X = merged_df[features]
    y = merged_df["co2"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = abs(y_test - y_pred)
    mean_error = error.mean()
    std_error = error.std()
    # Define threshold: mean + 3 * std (99.7% confidence interval under Gaussian assumption)
    failure_threshold = mean_error + 3 * std_error
    failure_flags = (error > failure_threshold).astype(int)

    results = X_test.copy()
    results["actual_co2"] = y_test
    results["predicted_co2"] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"📐 Regression Metrics:")
    print(f"   MAE  (Mean Absolute Error)      : {mae:.2f} ppm")
    print(f"   MSE  (Mean Squared Error)       : {mse:.2f} ppm²")
    print(f"   RMSE (Root Mean Squared Error)  : {rmse:.2f} ppm")
    print(f"   R²   (Coefficient of Determination): {r2:.4f}")

##############################
# FUNZIONI DI SUPPORTO
##############################

def assign_to_cluster(kmeans_model, scaler_model, clustering_assembler, sensor_data):
    # Aggiunge una colonna di default per la varianza del CO2
    sensor_data["CO2_variance"] = 0.0  
    assembled_sensor_data = clustering_assembler.transform(sensor_data)
    scaled_sensor_data = scaler_model.transform(assembled_sensor_data)
    cluster = kmeans_model.predict(scaled_sensor_data)
    return cluster[0], scaled_sensor_data

def train_regression_for_cluster(cluster_id, cluster_ranges, merged_df):
    """
    Addestra un modello di regressione lineare sul cluster specificato.
    Qui useremo i dati reali (merged_df deve essere il dataset reale).
    """
    # Per ottenere i range per il cluster: si possono usare i dati simulati
    # e poi applicare il filtro sui dati reali
    cluster_range = cluster_ranges[cluster_ranges["prediction"] == cluster_id].iloc[0]
    
    cluster_data = merged_df[
        (merged_df["temperature"] >= cluster_range["min_temperature"]) &
        (merged_df["temperature"] <= cluster_range["max_temperature"]) &
        (merged_df["volume"] >= cluster_range["min_volume"]) &
        (merged_df["volume"] <= cluster_range["max_volume"]) &
        (merged_df["ventilation rate"] >= cluster_range["min_ventilation"]) &
        (merged_df["ventilation rate"] <= cluster_range["max_ventilation"]) &
        (merged_df["humidity"] >= cluster_range["min_humidity"]) &
        (merged_df["humidity"] <= cluster_range["max_humidity"])
    ]
    
    if cluster_data.empty:
        print(f"Avviso: nessun dato reale per il cluster {cluster_id}")
        return None

    X = cluster_data[["temperature", "humidity", "ventilation rate", "volume"]]
    y = cluster_data["co2"]
    
    model = LinearRegression()
    model.fit(X, y)
    return model

def validate_sensor_data(lr_model, sensor_data, assembler):
    subfeatures_df = sensor_data[["temperature", "humidity", "ventilation rate", "volume"]]
    assembled_sensor_data = assembler.transform(subfeatures_df)
    X = pd.DataFrame(assembled_sensor_data, columns=["temperature", "humidity", "ventilation rate", "volume"])  
    predictions = lr_model.predict(X)
    
    sensor_data["prediction"] = predictions
    sensor_data["error"] = sensor_data["co2"] - predictions
    return sensor_data

def process_sensor_data(sensor_data, kmeans_model, scaler, assembler, cluster_ranges, real_df):
    """
    Assegna il dato di un sensore al cluster tramite i modelli addestrati (pretraining sui dati simulati)
    e poi, utilizzando i dati reali, allena (o fine-tuna) il modello di regressione per quel cluster.
    """
    sensor_cluster_id, scaled_sensor_data = assign_to_cluster(kmeans_model, scaler, assembler, sensor_data)
    print(f"Sensor data is assigned to cluster: {sensor_cluster_id}")
    
    # Fine-tuning: addestra il modello di regressione usando solo i dati reali
    lr_model = train_regression_for_cluster(sensor_cluster_id, cluster_ranges, real_df)
    if lr_model is None:
        print("Non è possibile addestrare il modello per questo cluster.")
        return sensor_data
    
    # Creiamo un assembler per la validazione sui dati reali
    assembler_validate = ColumnTransformer(
        transformers=[
            ("features", FunctionTransformer(lambda x: x, validate=False), 
             ["temperature", "humidity", "ventilation rate", "volume"]) 
        ]
    )
    assembler_validate.fit(real_df)
    validation_results = validate_sensor_data(lr_model, sensor_data, assembler_validate)
    return validation_results

##############################
# 1. CARICAMENTO DEI DATI
##############################

# Carica dati reali e simulati
input_file_path_real = "../decay-real/CO2_decay_filtered.csv"
data_df_real = pd.read_csv(input_file_path_real, delimiter=";")
merged_df_real = data_df_real[["temperature", "co2", "humidity", "volume", "ventilation rate"]].round(2)

input_file_path_sim = "../decay-simulated/CO2_decay_filtered.csv"
data_df_sim = pd.read_csv(input_file_path_sim, delimiter=";")
merged_df_sim = data_df_sim[["temperature", "co2", "humidity", "volume", "ventilation rate"]].round(2)

merged_df = pd.concat([merged_df_real, merged_df_sim], ignore_index=True)

##############################
# 2. PRETRAINING (SU DATI SIMULATI)
##############################

# Calcolo della varianza (filtraggio per varianza CO2 tra 0 e 20)
variance_df = merged_df_sim.groupby(["temperature", "humidity", "ventilation rate", "volume"]).agg(
    CO2_variance=("co2", "var")
).dropna().reset_index()

variance_df = variance_df[
    (variance_df["CO2_variance"] > 0) &
    (variance_df["CO2_variance"] < 20) &
    (variance_df["temperature"] > 20) & (variance_df["temperature"] < 25) &
    (variance_df["humidity"] > 20) & (variance_df["humidity"] < 80) &
    (variance_df["volume"] > 55) & (variance_df["volume"] < 75)
]

# Aggiungiamo la feature CO2_variance e definiamo le feature da usare
feature_columns = ["temperature", "humidity", "ventilation rate", "volume", "CO2_variance"]
variance_df = variance_df.astype(float)

assembler = ColumnTransformer(
    transformers=[
        ("features", FunctionTransformer(lambda x: x, validate=False), feature_columns)
    ]
)

# Applichiamo la trasformazione
assembled_df = variance_df.copy()
assembled_df["features"] = list(assembler.fit_transform(variance_df))

# Standardizzazione
features_matrix = np.vstack(assembled_df["features"])
scaler_model = StandardScaler()
scaled_features = scaler_model.fit_transform(features_matrix)
assembled_df["scaledFeatures"] = list(scaled_features)

# PCA e clustering gerarchico (visualizzazione)
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(scaled_features)
linkage_matrix = linkage(reduced_data, method='ward')

plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.savefig("dendrogram.png")
print("Dendrogram saved")

# Applica KMeans (usando solo dati simulati)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_model = kmeans.fit(scaled_features)
assembled_df["prediction"] = kmeans_model.predict(scaled_features)

# Creiamo il DataFrame finale per il clustering
final_df = assembled_df[["temperature", "humidity", "ventilation rate", "volume", "CO2_variance", "prediction"]]

# Estraiamo gli intervalli per ciascun cluster (cluster ranges) dai dati simulati
cluster_ranges = final_df.groupby("prediction").agg(
    min_temperature=("temperature", "min"),
    max_temperature=("temperature", "max"),
    min_volume=("volume", "min"),
    max_volume=("volume", "max"),
    min_ventilation=("ventilation rate", "min"),
    max_ventilation=("ventilation rate", "max"),
    min_humidity=("humidity", "min"),
    max_humidity=("humidity", "max")
).reset_index()

print("cluster_ranges:")
print(cluster_ranges)

lr_metrics(merged_df)

# Salviamo i dati clusterizzati (opzionale)
output_path = "output_low_variance_clusters.csv"
variance_df.to_csv(output_path, index=False)

##############################
# 3. FINE-TUNING (SU DATI REALI)
##############################

# Per il fine-tuning addestriamo i modelli di regressione per cluster usando solo i dati reali.
# (All'interno della funzione process_sensor_data viene richiamata train_regression_for_cluster,
#  che qui utilizza merged_df_real, passato come argomento.)

##############################
# 4. VALIDAZIONE SU DATI DI SENSORI
##############################

# Esempio di dati sensore
sensor_data_1 = pd.DataFrame([{
    "temperature": 22.35, "humidity": 43.5, 
    "ventilation rate": 0.25, "volume": 65.0, "co2": 1150.0
}])

print("\n##############################Start-Linear regression##############################")
print("\n########Sensor 1:\n")
print(sensor_data_1)

# Passiamo come "merged_df" i dati reali per il fine-tuning
validation_results_1 = process_sensor_data(
    sensor_data=sensor_data_1,
    kmeans_model=kmeans_model,
    scaler=scaler_model,
    assembler=assembler,
    cluster_ranges=cluster_ranges,
    real_df=merged_df_real  # Solo dati reali per fine-tuning
)

print(validation_results_1)

sensor_data_2 = pd.DataFrame([{
    "temperature": 22.35, "humidity": 43.5, 
    "ventilation rate": 0.25, "volume": 65.0, "co2": 5000.0
}])
print("\n########Sensor 2:\n")
print(sensor_data_2)

validation_results_2 = process_sensor_data(
    sensor_data=sensor_data_2,
    kmeans_model=kmeans_model,
    scaler=scaler_model,
    assembler=assembler,
    cluster_ranges=cluster_ranges,
    real_df=merged_df_real  # Fine-tuning su dati reali
)

print(validation_results_2)
print("\n##############################Stop-Linear regression##############################")
##############################
# 5. VISUALIZZAZIONE
##############################

df_clusters = pd.read_csv(output_path)
df_clusters = df_clusters[["temperature", "humidity", "volume", "CO2_variance"]]
# Se volessimo visualizzare un cluster specifico, ad esempio quello con prediction == 3
# (Notare che qui df_clusters non contiene la colonna "prediction", ma possiamo usare final_df se necessario)
filtered_df = final_df[final_df["prediction"] == 3]
#print(filtered_df)
#print(f"Number of rows: {len(filtered_df)}")

# Pairplot
sns.pairplot(
    final_df,
    vars=["temperature", "humidity", "volume", "CO2_variance"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
plt.savefig("cluster_pairplot.png")
print("Pairplot saved")

# Plot 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    final_df["temperature"],
    final_df["humidity"],
    final_df["volume"],
    c=final_df["prediction"],
    cmap="tab10",
    s=50
)
ax.set_xlabel("temperature")
ax.set_ylabel("humidity")
ax.set_zlabel("volume")
ax.set_title("3D Cluster Visualization")
plt.savefig("cluster_3d_plot.png")
print("3D plot saved")