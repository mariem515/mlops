import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ===== 2. Charger les données =====
df = pd.read_csv("churn_data.csv")
y = df["churn"]


# ===== 3. Fonction : préparer les données =====
def preparer(colonnes):
    X = pd.get_dummies(df[colonnes], drop_first=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ===== 4. Fonction : lancer un essai =====
def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    with mlflow.start_run(run_name=nom):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # --- Paramètres ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # --- Métriques ---
        mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
        mlflow.log_metric(
            "precision", precision_score(y_test, pred, zero_division=0)
        )
        mlflow.log_metric("recall", recall_score(y_test, pred, zero_division=0))
        mlflow.log_metric("f1", f1_score(y_test, pred, zero_division=0))

        # --- Artifact ---
        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        print(
            nom,
            "-> accuracy :",
            round(accuracy_score(y_test, pred), 3),
            "| f1 :",
            round(f1_score(y_test, pred, zero_division=0), 3),
        )


# ===== 5. Programme principal =====
mlflow.set_experiment("churn-prediction")
TOUT = ["age", "salaire", "abonnement", "anciennete", "consommation"]

lancer_essai("essai-1", 100, 5, TOUT)
lancer_essai("essai-2", 50, 3, TOUT)
lancer_essai("essai-3", 100, 5, ["age", "abonnement", "anciennete", "consommation"])
