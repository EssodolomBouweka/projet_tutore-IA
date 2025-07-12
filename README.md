# 🧩 Application d’Imputation de Données Manquantes

Cette application Streamlit permet d’importer un fichier de données (CSV ou Excel), de visualiser les valeurs manquantes, et de les imputer via différentes méthodes classiques ou basées sur l’apprentissage automatique (Random Forest, Régression, MICE...).

---

##  Fonctionnalités principales

-  Chargement de fichiers `.csv` ou `.xlsx`
-  Visualisation des valeurs manquantes (graphiques + tableau)
-  Choix de plusieurs méthodes d’imputation :
  - Suppression des lignes (avec valeurs manquantes)
  - Moyenne / Médiane / Mode
  - KNN (K-Nearest Neighbors)
  - Régression linéaire
  - Random Forest (avec regroupement facultatif)
  - MICE (Multiple Imputation by Chained Equations)
-  Gestion des variables catégorielles (encodage automatique)
-  Pagination pour grands jeux de données
-  Export des données imputées

---

##  Arborescence du projet

```bash
projet_tutore-IA/
├── app.py                         # Application principale Streamlit
├── requirements.txt              # Fichier des dépendances
├── imputers/
│   ├── __init__.py
│   ├── simple.py                 # Moyenne, médiane, mode, suppression
│   ├── knn.py                    # Imputation KNN
│   ├── rf.py                     # Imputation Random Forest
│   ├── regression.py             # Imputation par régression linéaire
│   ├── mice.py                   # Imputation MICE
├── utils/
│   ├── __init__.py
│   └── visualizer.py            # Visualisation des valeurs manquantes
├── models/                       # (Optionnel) modèles sauvegardés
└── README.md

Cloner le projet , installer les dépendances et exécuter la commende 'streamlit run app.py' et la magi va s'operer 

git clone https://github.com/EssodolomBouweka/projet_tutore-IA.git
cd projet_tutor--IA


# Activer l'environnement virtuel

python -m venv venv
venv\Scripts\activate        # Sur Windows
# ou
source venv/bin/activate     # Sur Mac/Linux

# installer les dépendances 

pip install -r requirements.txt

# Lancer l'application

streamlit run app.py
