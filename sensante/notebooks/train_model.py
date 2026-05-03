"""
Lab 2 : Entraîner et Sérialiser un Modèle
Projet SénSanté - ESP/UCAD - L2 GLSI
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# ÉTAPE 2 : Charger et préparer les données
# ============================================================
print("=" * 50)
print("ÉTAPE 2 : Chargement des données")
print("=" * 50)

df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# Encoder les variables catégoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")   # (500, 8)
print(f"Cible    : {y.shape}")    # (500,)

# ============================================================
# ÉTAPE 3 : Séparer entraînement et test
# ============================================================
print("\n" + "=" * 50)
print("ÉTAPE 3 : Séparation train/test")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% pour le test
    random_state=42,     # Reproductibilité
    stratify=y           # Proportions équilibrées
)

print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test         : {X_test.shape[0]} patients")

# ============================================================
# ÉTAPE 4 : Entraîner le modèle
# ============================================================
print("\n" + "=" * 50)
print("ÉTAPE 4 : Entraînement du modèle")
print("=" * 50)

model = RandomForestClassifier(
    n_estimators=100,    # 100 arbres de décision
    random_state=42      # Reproductibilité
)

model.fit(X_train, y_train)

print("Modèle entraîné !")
print(f"Nombre d'arbres  : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# ============================================================
# ÉTAPE 5 : Évaluer le modèle
# ============================================================
print("\n" + "=" * 50)
print("ÉTAPE 5 : Évaluation du modèle")
print("=" * 50)

y_pred = model.predict(X_test)

# Comparer les 10 premières prédictions
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prédiction':      y_pred[:10]
})
print("\nComparaison (10 premiers patients) :")
print(comparison.to_string(index=False))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nMatrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Visualisation (optionnelle)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SénSanté')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.close()
print("Figure sauvegardée dans figures/confusion_matrix.png")

# ============================================================
# ÉTAPE 6 : Sérialiser le modèle
# ============================================================
print("\n" + "=" * 50)
print("ÉTAPE 6 : Sérialisation du modèle")
print("=" * 50)

os.makedirs("models", exist_ok=True)

joblib.dump(model,       "models/model.pkl")
joblib.dump(le_sexe,     "models/encoder_sexe.pkl")
joblib.dump(le_region,   "models/encoder_region.pkl")
joblib.dump(feature_cols,"models/feature_cols.pkl")

size = os.path.getsize("models/model.pkl")
print(f"Modèle sauvegardé : models/model.pkl")
print(f"Taille            : {size / 1024:.1f} Ko")
print("Encodeurs et metadata sauvegardés.")

# ============================================================
# ÉTAPE 7 : Tester le modèle sérialisé
# ============================================================
print("\n" + "=" * 50)
print("ÉTAPE 7 : Test du modèle rechargé")
print("=" * 50)

model_loaded     = joblib.load("models/model.pkl")
le_sexe_loaded   = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Nouveau patient fictif
nouveau_patient = {
    'age':         28,
    'sexe':        'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux':        True,
    'fatigue':     True,
    'maux_tete':   True,
    'region':      'Dakar'
}

sexe_enc   = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas     = model_loaded.predict_proba([features])[0]
proba_max  = probas.max()

print(f"\n--- Résultat du pré-diagnostic ---")
print(f"Patient    : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité: {proba_max:.1%}")

print(f"\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:8s} : {proba:.1%}  {bar}")

# ============================================================
# EXERCICE 1 : Importance des features
# ============================================================
print("\n" + "=" * 50)
print("EXERCICE 1 : Importance des features")
print("=" * 50)

importances = model.feature_importances_
print("\nFeatures classées par importance :")
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    bar = '█' * int(imp * 50)
    print(f"  {name:20s} : {imp:.3f}  {bar}")

# ============================================================
# EXERCICE 2 : 3 patients fictifs
# ============================================================
print("\n" + "=" * 50)
print("EXERCICE 2 : Prédictions sur 3 patients fictifs")
print("=" * 50)

patients_fictifs = [
    {'age': 12, 'sexe': 'M', 'temperature': 36.8, 'tension_sys': 100, 'toux': False, 'fatigue': False, 'maux_tete': False, 'region': 'Dakar'},
    {'age': 35, 'sexe': 'F', 'temperature': 40.2, 'tension_sys': 130, 'toux': True,  'fatigue': True,  'maux_tete': True,  'region': 'Thies'},
    {'age': 68, 'sexe': 'M', 'temperature': 38.5, 'tension_sys': 145, 'toux': True,  'fatigue': True,  'maux_tete': False, 'region': 'Kaolack'},
]
descriptions = [
    "Enfant 12 ans, sans symptômes",
    "Adulte 35 ans, forte fièvre",
    "Personne âgée 68 ans, toux + fatigue",
]

for p, desc in zip(patients_fictifs, descriptions):
    s_enc = le_sexe_loaded.transform([p['sexe']])[0]
    r_enc = le_region_loaded.transform([p['region']])[0]
    f = [p['age'], s_enc, p['temperature'], p['tension_sys'],
         int(p['toux']), int(p['fatigue']), int(p['maux_tete']), r_enc]
    diag  = model_loaded.predict([f])[0]
    proba = model_loaded.predict_proba([f])[0].max()
    print(f"  [{desc}] → {diag} ({proba:.1%})")

print("\n✅ Lab 2 terminé avec succès !")
print("   Prochaine étape : Lab 3 - API FastAPI (tag v2)")
