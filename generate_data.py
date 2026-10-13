import pandas as pd
import numpy as np

# Pour obtenir les mêmes données à chaque exécution
np.random.seed(42)

# Nombre de clients
n = 500

# Création du dataset
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "salaire": np.random.randint(1200, 8000, n),
    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"], n
    ),
    "anciennete": np.random.randint(0, 15, n),
    "consommation": np.random.randint(0, 100, n)
})

# Création de la target : churn
data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int)

# Sauvegarde dans un fichier CSV
data.to_csv("churn_data.csv", index=False)

# Vérification
print(data.head())
print(f"Taux de churn : {data['churn'].mean():.1%}")