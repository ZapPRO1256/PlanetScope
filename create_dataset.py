import pandas as pd
import numpy as np

# ----------------------
# 1. Функції для уніфікації кожної бази
# ----------------------

def unify_kepler(df):
    df = df.rename(columns={
        'kepoi_name':'planet_name',
        'koi_period':'orbital_period',
        'koi_duration':'transit_duration',
        'koi_depth':'transit_depth',
        'koi_prad':'planet_radius',
        'koi_insol':'insolation',
        'koi_teq':'eq_temperature',
        'koi_eccen':'eccentricity',
        'koi_incl':'inclination',
        'koi_ror':'planet_star_radius_ratio',
        'koi_sma':'semi_major_axis',
        'koi_impact':'impact_param',
        'koi_steff':'stellar_teff',
        'koi_srad':'stellar_radius',
        'koi_smass':'stellar_mass',
        'koi_slogg':'stellar_logg',
        'koi_smet':'stellar_metal',
        'koi_sage':'stellar_age'
    })

    # Лишаємо тільки CONFIRMED та FALSE POSITIVE (ігноруємо CANDIDATE)
    df = df[df['koi_disposition'] != "CANDIDATE"]

    # Label: CONFIRMED = 1, FALSE POSITIVE = 0
    df['label'] = df['koi_disposition'].apply(lambda x: 1 if x == "CONFIRMED" else 0)

    final_cols = ['planet_name','orbital_period','transit_duration','transit_depth','planet_radius',
                  'insolation','eq_temperature','eccentricity','inclination','planet_star_radius_ratio',
                  'semi_major_axis','impact_param','stellar_teff','stellar_radius','stellar_mass',
                  'stellar_logg','stellar_metal','stellar_age','label']
    return df[final_cols]

def unify_k2(df):
    df = df.rename(columns={
        'pl_name':'planet_name',
        'pl_orbper':'orbital_period',
        'pl_trandur':'transit_duration',
        'pl_trandep':'transit_depth',
        'pl_rade':'planet_radius',
        'pl_insol':'insolation',
        'pl_eqt':'eq_temperature',
        'pl_orbeccen':'eccentricity',
        'pl_orbincl':'inclination',
        'pl_ratror':'planet_star_radius_ratio',
        'pl_orbsmax':'semi_major_axis',
        'pl_imppar':'impact_param',
        'st_teff':'stellar_teff',
        'st_rad':'stellar_radius',
        'st_mass':'stellar_mass',
        'st_logg':'stellar_logg',
        'st_met':'stellar_metal',
        'st_age':'stellar_age'
    })

    missing_cols = ['eccentricity','inclination','planet_star_radius_ratio','semi_major_axis','impact_param']
    for col in missing_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Ігноруємо CANDIDATE
    df = df[df['disposition'] != "CANDIDATE"]

    df['label'] = df['disposition'].apply(lambda x: 1 if x == "CONFIRMED" else 0)

    final_cols = ['planet_name','orbital_period','transit_duration','transit_depth','planet_radius',
                  'insolation','eq_temperature','eccentricity','inclination','planet_star_radius_ratio',
                  'semi_major_axis','impact_param','stellar_teff','stellar_radius','stellar_mass',
                  'stellar_logg','stellar_metal','stellar_age','label']
    return df[final_cols]

def unify_toi(df):
    df = df.rename(columns={
        'toi':'planet_name',
        'pl_orbper':'orbital_period',
        'pl_trandurh':'transit_duration',
        'pl_trandep':'transit_depth',
        'pl_rade':'planet_radius',
        'pl_insol':'insolation',
        'pl_eqt':'eq_temperature',
        'st_teff':'stellar_teff',
        'st_rad':'stellar_radius',
        'st_logg':'stellar_logg'
    })

    missing_cols = ['eccentricity','inclination','planet_star_radius_ratio',
                    'semi_major_axis','impact_param','stellar_mass','stellar_metal','stellar_age']
    for col in missing_cols:
        df[col] = np.nan

    df = df[df['tfopwg_disp'] != "PC"]

    df['label'] = df['tfopwg_disp'].apply(lambda x: 1 if (x == "CP" or x=="KP") else 0)

    final_cols = ['planet_name','orbital_period','transit_duration','transit_depth','planet_radius',
                  'insolation','eq_temperature','eccentricity','inclination','planet_star_radius_ratio',
                  'semi_major_axis','impact_param','stellar_teff','stellar_radius','stellar_mass',
                  'stellar_logg','stellar_metal','stellar_age','label']
    return df[final_cols]

# ----------------------
# 2. Зчитування CSV
# ----------------------

kepler = pd.read_csv('cumulative_2025.09.27_03.08.53.csv', comment='#')
k2 = pd.read_csv('k2pandc_2025.09.27_03.10.49.csv', comment='#')
toi = pd.read_csv('TOI_2025.09.27_03.10.05.csv', comment='#')

# ----------------------
# 3. Уніфікація
# ----------------------
df_kepler = unify_kepler(kepler)
df_k2 = unify_k2(k2)
df_toi = unify_toi(toi)

# ----------------------
# 4. Об'єднання
# ----------------------
df_all = pd.concat([df_kepler, df_k2, df_toi], ignore_index=True)

# ----------------------
# 5. Збереження фінального датасету
# ----------------------
df_all.to_csv('yea.csv', index=False)
print("Об'єднаний датасет створено: unified_exoplanets_dataset.csv")
