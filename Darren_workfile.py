import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = 'C:/Users/darre/OneDrive/Documents/Horizon Zero Dawn/GACTT_RESULTS_ANONYMIZED.csv'
df_uploaded = pd.read_csv(file_path)

# Sélection et préparation des colonnes nécessaires
df_cleaned = df_uploaded[['Gender', 'How many cups of coffee do you typically drink per day?', 'What is your age?']].copy()

# Renommer les colonnes pour faciliter l'utilisation
df_cleaned.rename(columns={
    'Gender': 'Sexe',
    'How many cups of coffee do you typically drink per day?': 'Consommation Quotidienne de Café',
    'What is your age?': 'Âge'
}, inplace=True)

def nettoyer_conso(x):
    try:
        return float(x)
    except ValueError:
        return 0
df_cleaned['Consommation Quotidienne de Café'] = df_cleaned['Consommation Quotidienne de Café'].apply(nettoyer_conso)

le_sexe = LabelEncoder()
le_age = LabelEncoder()

df_cleaned['Sexe'] = le_sexe.fit_transform(df_cleaned['Sexe'].astype(str))
df_cleaned['Âge'] = le_age.fit_transform(df_cleaned['Âge'].astype(str))

df_cleaned.dropna(subset=['Sexe', 'Âge'], inplace=True)

df_grouped = df_cleaned.groupby(['Sexe', 'Âge'])['Consommation Quotidienne de Café'].mean().reset_index()
df_grouped = df_cleaned.groupby(['Sexe', 'Âge'])['Consommation Quotidienne de Café'].mean().reset_index()

df_grouped['Sexe-Âge'] = df_grouped.apply(
    lambda row: f"{le_sexe.inverse_transform([int(row['Sexe'])])[0]}-{le_age.inverse_transform([int(row['Âge'])])[0]}",
    axis=1
)

plt.figure(figsize=(10, 6))
sns.barplot(y='Sexe-Âge', x='Consommation Quotidienne de Café', data=df_grouped, orient='h')
plt.title('Consommation Moyenne de Café par Jour selon le Sexe et la Tranche d\'Âge')
plt.xlabel('Consommation Moyenne de Café (tasses par jour)')
plt.ylabel('Sexe et Tranche d\'Âge')
plt.show()
