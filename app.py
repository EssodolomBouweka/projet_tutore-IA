import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from imputers import simple, knn, rf, regression, mice
from utils.visualizer import plot_missing_values

# Lecture dynamique selon le type de fichier
def load_data(file):
    file_ext = file.name.split('.')[-1].lower()

    try:
        if file_ext == "csv":
            return pd.read_csv(file)
        elif file_ext in ["xlsx", "xls"]:
            return pd.read_excel(file)
        elif file_ext == "json":
            return pd.read_json(file)
        elif file_ext == "parquet":
            return pd.read_parquet(file)
        elif file_ext == "txt":
            return pd.read_csv(file, delimiter="\t")  # tabulation
        else:
            st.error("❌ Format de fichier non pris en charge.")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier : {e}")
        return None

# Affichage avec pagination
def display_paginated_dataframe(df, page_size=50, key=None):
    total_rows = df.shape[0]
    total_pages = (total_rows // page_size) + (1 if total_rows % page_size > 0 else 0)

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=key
    )
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    st.write(f"Afficher les lignes {start_idx + 1} à {end_idx} sur {total_rows}")
    st.dataframe(df.iloc[start_idx:end_idx])

# Encodage des colonnes catégorielles
def encode_categorical(df):
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype(str).fillna('missing')
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded

# Suggestions dynamiques
def suggest_imputation_methods(df):
    total_cells = df.size
    total_missing = df.isna().sum().sum()
    missing_ratio = total_missing / total_cells if total_cells > 0 else 0

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    num_missing = df[num_cols].isna().sum().sum() if len(num_cols) > 0 else 0
    cat_missing = df[cat_cols].isna().sum().sum() if len(cat_cols) > 0 else 0

    suggestions = []

    if missing_ratio == 0:
        suggestions.append("✅ Aucune valeur manquante détectée.")
    else:
        if missing_ratio < 0.05:
            suggestions.append("ℹ️ Faible taux de valeurs manquantes (<5%) → suppression envisageable.")
        else:
            suggestions.append("⚠️ Taux élevé de valeurs manquantes → privilégiez l’imputation.")

        if num_missing > cat_missing:
            suggestions.append("🔢 Colonnes numériques majoritairement manquantes → Moyenne, Médiane, Régression, Random Forest recommandées.")
        if cat_missing > 0:
            suggestions.append("🔤 Colonnes catégorielles manquantes → Mode, KNN ou MICE recommandés.")

    return suggestions

# Configuration Streamlit
st.set_page_config(page_title="Imputation de Données", layout="wide")
st.title("🧩 Application d’Imputation de Données Manquantes")

# Chargement du fichier
uploaded_file = st.file_uploader(
    "📂 Chargez votre fichier de données",
    type=["csv", "xlsx", "xls", "txt", "json", "parquet"]
)

# Méthodes d'imputation
imputation_methods = {
    "Suppression des lignes": simple.delete_rows,
    "Moyenne": simple.impute_mean,
    "Médiane": simple.impute_median,
    "Mode": simple.impute_mode,
    "KNN": knn.impute_knn,
    "Régression linéaire": regression.impute_with_linear_regression,
    "Random Forest": rf.impute_rf,
    "MICE": mice.impute_mice,
}
ml_methods = ["Random Forest", "Régression linéaire", "KNN"]

# Traitement principal
if uploaded_file:
    df = load_data(uploaded_file)
    if df is None:
        st.stop()

    st.write("📋 **Aperçu des données importées (pagination disponible)**")
    display_paginated_dataframe(df, key="original_data")

    total_missing = df.isna().sum().sum()
    st.markdown(f"🧮 **Nombre total de valeurs manquantes** : `{total_missing}`")

    st.subheader("💡 Suggestions de méthode d’imputation")
    for s in suggest_imputation_methods(df):
        st.info(s)

    st.subheader("📊 Visualisation des valeurs manquantes")
    st.pyplot(plot_missing_values(df))

    method_name = st.selectbox("⚙️ Choisissez une méthode d’imputation :", list(imputation_methods.keys()))

    if st.button("🚀 Lancer l’imputation"):
        try:
            impute_function = imputation_methods[method_name]

            if method_name in ml_methods:
                df_to_impute = encode_categorical(df)
            else:
                df_to_impute = df

            if method_name == "Random Forest":
                with st.expander("🔧 Paramètres avancés Random Forest"):
                    group_col = st.selectbox("Colonne de regroupement (optionnelle)", [None] + list(df.columns))
                    df_imputed = impute_function(df_to_impute, group_col=group_col if group_col != "None" else None)
            else:
                df_imputed = impute_function(df_to_impute)

            st.success(f"✅ Imputation avec la méthode : **{method_name}**")
            st.write("📋 **Aperçu des données imputées (pagination disponible)**")
            display_paginated_dataframe(df_imputed, key="imputed_data")

            st.download_button(
                label="📥 Télécharger les données imputées",
                data=df_imputed.to_csv(index=False),
                file_name="data_imputed.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {str(e)}")
