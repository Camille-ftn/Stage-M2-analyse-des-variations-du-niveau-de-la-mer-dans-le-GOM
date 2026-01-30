# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:33:51 2025

@author: camil
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import cartopy.crs as ccrs 
import xarray as xr
import glob
import cartopy.feature  as cfeature
import datetime
import os 
import statsmodels.api as sm
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, t
from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL
from haversine import haversine, Unit
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import pearsonr
from scipy.signal import detrend
from scipy.fftpack import fft, fftfreq
from pyts.decomposition import SingularSpectrumAnalysis
import re
from scipy.interpolate import interp1d
from scipy.special import comb
import seaborn as sns
import matplotlib.colors as colors
import scipy.signal as sig
from eofs.standard import Eof
from pymssa import MSSA
from statsmodels.tsa.seasonal import seasonal_decompose
from PyEMD import EMD

# Charger le fichier .gpkg dans un GeoDataFrame
gdf = gpd.read_file(r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\courbes de niveaux .gpkg")
if 'ELEV' in gdf.columns:
    gdf400 = gdf[gdf['ELEV'] <= -400]  # Filtrer les courbes de niveau à -200 m ou moins
else:
    print("La colonne 'elevation' n'a pas été trouvée dans le fichier .gpkg")
if 'ELEV' in gdf.columns:
    gdf200 = gdf[gdf['ELEV'] == -200]

# Chemin vers les fichiers .nc
path = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\SEANEO\*.nc"

# Utilisation de glob pour obtenir tous les fichiers .nc
files = glob.glob(path)

#%%--------marégraphes--------
# Chemin vers le dossier contenant les fichiers .txt
dossier = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\PSMSL\*.txt"

# Dictionnaire pour stocker les DataFrames
dataframes = {}

# Parcourir tous les fichiers .txt du dossier
for fichier in glob.glob(dossier):
    # Extraire le nom du fichier sans le chemin ni l'extension
    nom_fichier =  os.path.splitext(os.path.basename(fichier))[0]
    
    # Lire le fichier dans un DataFrame (adapte le séparateur si nécessaire)
    df = pd.read_csv(fichier, sep=";", comment='#',header=None)  # Exemple avec tabulation, change 'sep' si besoin
    # Supprimer les deux dernières colonnes
    df = df.drop(columns=[2, 3])
    # Renommer les colonnes
    df = df.rename(columns={0: 'temps', 1: 'hauteur d\'eau en mm'})
    # Convertir 'temps' en format date
    df['temps'] = pd.to_datetime(df['temps'], format='%Y') + pd.to_timedelta((df['temps'] % 1) * 365.25 * 24 * 3600, unit='s')
    df["temps"] = df["temps"].dt.round("S")
    df['hauteur d\'eau en mm'] = df['hauteur d\'eau en mm'].replace(-99999, np.nan)
    # Interpolation pour DAUPHIN ISLAND, si la station est présente
    if nom_fichier == "DAUPHIN ISLAND":
        # Filtrer les données pour 2004 et 2006
        df_2004 = df[df['temps'].dt.year == 2004]
        df_2006 = df[df['temps'].dt.year == 2006]
        
        if not df_2004.empty and not df_2006.empty:
            # Interpolation des valeurs manquantes pour 2005
            # Créer un DataFrame pour 2005 basé sur la linéarité entre 2004 et 2006
            start_value = df_2004['hauteur d\'eau en mm'].iloc[0]
            end_value = df_2006['hauteur d\'eau en mm'].iloc[0]
            
            # Créer une série de dates pour l'année 2005 (12 mois)
            dates_2005 = pd.date_range(start='2005-01-01', end='2005-12-31', freq='M')
            
            # Interpolation linéaire entre la première valeur de 2004 et la première valeur de 2006
            hauteur_2005 = np.linspace(start_value, end_value, len(dates_2005))
            
            # Créer un DataFrame avec les dates et les hauteurs interpolées
            df_2005 = pd.DataFrame({'temps': dates_2005, 'hauteur d\'eau en mm': hauteur_2005})
            
            # Ajouter les données interpolées pour 2005 au DataFrame original
            df = pd.concat([df, df_2005]).sort_values('temps')
    # Ajouter au dictionnaire
    dataframes[nom_fichier] = df 

# Définir la période d'étude
periode_debut = pd.Timestamp("2002-01-01")
periode_fin = pd.Timestamp("2021-06-30")

# Vérifier la disponibilité des données
for nom_fichier, df in dataframes.items():
    # Filtrer la période d'étude
    df_periode = df[(df["temps"] >= periode_debut) & (df["temps"] <= periode_fin)]
    
    # Calculer le pourcentage de valeurs non NaN
    taux_disponibilite = df_periode["hauteur d'eau en mm"].notna().mean() * 100

    # Vérifier si c'est supérieur à 80 %
    if taux_disponibilite >= 80:
        print(f"{nom_fichier} : ✅ {taux_disponibilite:.2f}% des données disponibles")
    else:
        print(f"{nom_fichier} : ❌ Seulement {taux_disponibilite:.2f}% des données disponibles")

coordonnees = {
    "VERACRUZ": {"lat": 19.18, "lon": -96.12},
    "PORT ISABEL": {"lat": 26.07, "lon": -97.22},
    "ROCKPORT": {"lat": 28.02, "lon": -97.05},
    "FREEPORT": {"lat": 28.95, "lon": -95.32},
    "GALVESTON I": {"lat": 29.32, "lon": -94.80},
    "GALVESTON II": {"lat": 29.28, "lon": -94.78},
    "GRAND ISLE": {"lat": 29.28, "lon": -89.97},
    "DAUPHIN ISLAND": {"lat": 30.25, "lon": -88.07},
    "PENSACOLA": {"lat": 30.40, "lon": -87.22},
    "APALACHICOLA": {"lat": 29.72, "lon": -84.98},
    "CEDAR KEY II": {"lat": 29.13, "lon": -83.03},
    "ST. PETERSBURG": {"lat": 27.77, "lon": -82.62},
    "FORT MYERS": {"lat": 26.65, "lon": -81.87},
    "NAPLES": {"lat": 26.13, "lon": -81.80},
    "KEY WEST": {"lat": 24.55, "lon": -81.80}
}
# Ajouter les coordonnées à chaque DataFrame
for station, df in dataframes.items():
    lat = coordonnees[station]["lat"]
    lon = coordonnees[station]["lon"]

    # Ajouter des colonnes pour latitude et longitude
    df["lat"] = lat
    df["lon"] = lon
    

#%%---------seanoe + XTRACK---------
tref = datetime.datetime(1950, 1, 1)
tstart = datetime.datetime(1992, 1, 1)
tend = datetime.datetime(2022, 12, 31)

# Chemin vers les fichiers .nc seanoe
path_s = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\SEANEO\*.nc"
seanoe = glob.glob(path_s)

#Chemin vers les fichiers .nc XTRACK
path_x= r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\GOM_XTRACK_LP2\*.nc"
xtrack = glob.glob(path_x)

#----------def---------

def find_nearest_sla_points(files, coordinates, max_distance_km):
    """
    Trouve les points SLA les plus proches des stations marégraphiques en utilisant la distance géodésique,
    en filtrant uniquement les points situés dans le Golfe du Mexique.

    Paramètres :
    - files (list) : Liste des fichiers NetCDF contenant les données SLA.
    - coordinates (dict) : Dictionnaire des coordonnées des marégraphes {"station": {"lat": val, "lon": val}}.
    - max_distance_km (float) : Distance maximale de recherche en km.

    Retourne :
    - nearest_points (dict) : Dictionnaire avec les stations ayant un point SLA proche {"station": {"latitude": val, "longitude": val}}.
    """

    nearest_points = {}

    # Définition des limites géographiques du Golfe du Mexique
    lat_min, lat_max = 18, 31  # Entre le sud du Mexique et la Floride
    lon_min, lon_max = -98, -80  # De la côte mexicaine à la Floride

    for file in files:
        ds = xr.open_dataset(file, decode_times=False)

        # Extraire les latitudes et longitudes
        latitudes = ds['lat'].values
        longitudes = ds['lon'].values

        # Filtrer pour ne garder que les points SLA situés dans le Golfe du Mexique
        latitudes = latitudes[(latitudes >= lat_min) & (latitudes <= lat_max)]
        longitudes = longitudes[(longitudes >= lon_min) & (longitudes <= lon_max)]

        # Vérifier s'il reste des points SLA après le filtrage
        if len(latitudes) == 0 or len(longitudes) == 0:
            #print("⚠️ Aucun point SLA dans la zone du Golfe du Mexique.")
            continue

        # Créer une liste de toutes les coordonnées SLA disponibles après filtrage
        grid_points = [(lat, lon) for lat in latitudes for lon in longitudes]

        # Itérer sur chaque station marégraphique
        for station, coord in coordinates.items():
            lat_tg, lon_tg = coord["lat"], coord["lon"]

            # Vérifier si la station est bien située dans le Golfe du Mexique
            # if not (lat_min <= lat_tg <= lat_max and lon_min <= lon_tg <= lon_max):
            #     #print(f"⛔ Station {station} hors du Golfe du Mexique, ignorée.")
            #     continue  # Passer à la station suivante

            # Initialiser la distance minimale très grande et la position du meilleur point
            min_distance = 60
            best_point = None

            # Trouver le point SLA le plus proche
            for lat, lon in grid_points:
                distance = haversine((lat_tg, lon_tg), (lat, lon), unit=Unit.KILOMETERS)

                # Vérifier si la station est déjà stockée
                if station in nearest_points:
                    prev_distance = nearest_points[station]['distance']
                    if distance >= prev_distance:
                        continue  # Si la distance est plus grande, on garde l'ancien point

                # Sinon, on met à jour avec le point actuel
                nearest_points[station] = {"latitude": lat, "longitude": lon, "distance": distance}
                print(f"✅ Station {station} : Point SLA -> Lat: {lat}, Lon: {lon}, Distance: {distance:.2f} km")

        ds.close()

    return nearest_points
def apply_threshold_and_remove_neighbors(dac, sigma_multiple=3):
    """
    Applique un filtre de seuil basé sur sigma et élimine les voisins adjacents des valeurs aberrantes.
    
    Parameters:
    - dac: données de l'altimétrie (de type numpy array)
    - sigma_multiple: multiple de l'écart-type (3 ou 4)
    
    Returns:
    - dac: les données après application du filtre de seuil et suppression des voisins
    """
    # Calcul de l'écart-type (σ) le long de la trace
    sigma = np.nanstd(dac, axis=0)  # Calcul de l'écart-type le long de l'axe des traces
    mean = np.nanmean(dac, axis=0)  # Moyenne des données le long de la trace
    
    # Appliquer le seuil de 3σ ou 4σ
    threshold_upper = mean + sigma_multiple * sigma
    threshold_lower = mean - sigma_multiple * sigma
    
    # Remplacer les valeurs qui dépassent ce seuil par NaN
    outliers = (dac > threshold_upper) | (dac < threshold_lower)
    
    # Marquer les valeurs aberrantes et leurs voisins
    # for i in range(1, len(dac)-1):
    #     if outliers[i].any():  # Si la valeur i est aberrante
    #         dac[i] = np.nan  # Supprimer la valeur aberrante
    #         dac[i-1] = np.nan  # Supprimer la valeur précédente
    #         dac[i+1] = np.nan  # Supprimer la valeur suivante

    # Après avoir éliminé les voisins des aberrations, on garde les autres valeurs.
    return dac


def plot_nearest_points_on_map(nearest_points):
    """
    Affiche les points SLA les plus proches des stations marégraphiques sur une carte.
    
    Paramètres :
    - nearest_points (dict) : Dictionnaire avec les stations et leurs points SLA proches.
    """
    # Créer la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Ajouter une carte de fond (ex : carte du monde)
    ax.set_extent([-100, -80, 17, 32], crs=ccrs.PlateCarree())  # Zone du Golfe du Mexique
    ax.coastlines(resolution='10m')  # Lignes côtières
    ax.gridlines(draw_labels=True)
    
    # Tracer les points SLA les plus proches
    lats = []
    lons = []
    for station, coords in nearest_points.items():
        lats.append(coords['latitude'])
        lons.append(coords['longitude'])
    
    # Convertir les listes en arrays pour plus de simplicité
    lats = np.array(lats)
    lons = np.array(lons)
    
    # Ajouter les points SLA sur la carte
    ax.scatter(lons, lats, color='red', marker='x', s=100, label='Points SLA', transform=ccrs.PlateCarree())
    
    # Ajouter les labels des stations
    for station, coords in nearest_points.items():
        ax.text(coords['longitude'] + 0.05, coords['latitude'] + 0.05, station, 
                color='blue', fontsize=8, transform=ccrs.PlateCarree())
    
    # Ajouter un titre
    plt.title('Points SLA les plus proches des stations marégraphiques')
    plt.legend(loc='upper left')

    # Afficher la carte
    plt.show()
    

def compute_trend_gmm(years_subset, sla_subset):
    """Effectue une régression par moindres carrés généralisés (GMM) et retourne la tendance, l'erreur, le R² et l'intervalle de confiance."""
    if len(years_subset) < 2:  # Besoin d'au moins deux points pour une tendance
        return None, None, None, None

    X = sm.add_constant(years_subset)  # Ajout d'une constante pour l'interception
    # Calcul des poids, ici un exemple simple avec l'inverse de la variance des erreurs
    weights = 1 / (sla_subset ** 2)  # Ex: Utilisation de la variance des résidus comme poids (hypothèse)

    # Régression des moindres carrés pondérés (WLS)
    model = sm.WLS(sla_subset, X, weights=weights).fit()

    # Tendance en mm/an
    trend = model.params[1] * 10**3  # Conversion en mm/an
    trend_error = model.bse[1] * 10**3  # Erreur standard en mm/an
    conf_interval = model.conf_int(alpha=0.05)[1] * 10**3  # Intervalle de confiance à 95% en mm/an
    r_squared = model.rsquared  # Coefficient de détermination

    # Retourner les résultats arrondis
    return round(trend, 2), round(trend_error, 2), conf_interval, round(r_squared, 3)


# Fonction pour trouver l'indice du point le plus proche en utilisant cKDTree
def find_nearest_idx(lat_nearest, lon_nearest, lat, lon,k):
    # Création de l'arbre k-d pour la recherche du voisin le plus proche
    tree = cKDTree(np.column_stack((lat, lon)))
    # Recherche du plus proche voisin
    dist, idx_spatial = tree.query([lat_nearest,lon_nearest], k=k)
    #print(f"Station lat={lat_nearest}, lon={lon_nearest} → Indices trouvés: {idx_spatial}")

    return dist,idx_spatial

def plot_trend(years_subset, sla_subset, dates_subset, label, color):
    if len(years_subset) > 1:
        # Ajustement linéaire pour la tendance (pour obtenir pente et son erreur)
        X_lin = sm.add_constant(years_subset)  # [1, t]
        model_lin = sm.OLS(sla_subset, X_lin).fit()
        trend_slope = round(model_lin.params[1], 2)  # pente (mm/an)
        trend_error = round(model_lin.bse[1], 2)      # erreur sur la pente

        # Ajustement quadratique pour obtenir l'accélération
        # Créer la matrice de design pour le modèle quadratique : [1, t, t^2]
        X_quad = np.column_stack((np.ones(len(years_subset)), years_subset, years_subset**2))
        model_quad = sm.OLS(sla_subset, X_quad).fit()
        # Le coefficient pour t^2
        a = model_quad.params[2]
        a_std = model_quad.bse[2]
        # L'accélération est 2 * a
        acceleration = round(2 * a, 2)
        acceleration_std = round(2 * a_std, 2)
        
        # Calcul de la tendance ajustée avec le modèle linéaire (pour simplifier le tracé)
        trend_line = np.polyval([model_lin.params[1], model_lin.params[0]], years_subset)

        # Tracé de la ligne de tendance
        plt.plot(dates_subset, trend_line, 
                 label=f"{label} : {trend_slope} ± {trend_error} mm/an, acc: {acceleration} ± {acceleration_std} mm/an²", 
                 color=color, linestyle='--')


def compute_sla_with_rolling_mean_x(xtrack, nearest_points_x,window_size,w):
    sla_time_series = {}

    for file in xtrack:
        ds = xr.open_dataset(file, decode_times=False)
        
        lat = ds['lat'].data
        lon = ds['lon'].data
        time = np.nanmean(ds['time'].data, axis=0)

        dtmission = 9.9156
        for t in range(1, len(time)):
            if np.isnan(time[t]):
                time[t] = time[t-1] + dtmission

        T = np.array([tref + datetime.timedelta(days=t) for t in time])
        ds = ds.assign_coords(time=("cycles_numbers", T))

        id1start = np.searchsorted(T, tstart, side="left")
        id1end = np.searchsorted(T, tend, side="right") - 1
        ds = ds.sel(cycles_numbers=slice(id1start, id1end))
        ds["sla_dac"] = ds["sla"] + ds["dac"]
        for station, coord in nearest_points_x.items():
        
            lat_nearest = coord["latitude"]
            lon_nearest = coord["longitude"]

            # Trouver les indices des 10 points les plus proches avec cKDTree
            dist, idx_spatial = find_nearest_idx(lat_nearest, lon_nearest, lat, lon, k=1)
    
            if np.any(dist == np.inf):  # Vérifier si aucun point valide n'a été trouvé
                #(f"Aucun point valide trouvé pour {station} dans {file}, skipping...")
                continue
            
            if station in sla_time_series:
                prev_dist = sla_time_series[station]['distance']
                if dist >= prev_dist:
                    continue
        
            # Extraire et moyenner la SLA sur les 10 points les plus proches
            sla_series = ds['sla'][idx_spatial, :].values
    
            # Extraire les coordonnées moyennes des points sélectionnés
            lat_selected = ds['lat'].isel(points_numbers=idx_spatial).values
            lon_selected = ds['lon'].isel(points_numbers=idx_spatial).values
              
            valid_indices = ~np.isnan(sla_series)
            sla_series = sla_series[valid_indices]  
            
            dates = ds['time'].isel(cycles_numbers=slice(id1start, id1end)).values
            dates = dates[valid_indices]
            dates = pd.to_datetime(dates)
        
            d=pd.DataFrame(dates)
            time_num=(d[0]-d[0].min()).dt.total_seconds().values
            
            X = sm.add_constant(time_num)
            model = sm.GLS(sla_series, X).fit()
            trend_removed = sla_series - model.fittedvalues  
            window_size_sla_x = w.get(station)
            ssa = SingularSpectrumAnalysis(window_size=window_size_sla_x , groups='auto')
            ssa_components = ssa.fit_transform(trend_removed.reshape(1, -1))
            num_modes = ssa_components.shape[1]  # Nombre de modes réellement disponibles
            
            # fig, axes = plt.subplots(num_modes, 1, figsize=(10, 2 * num_modes), sharex=True)
            
            # # S'assurer que axes est toujours une liste
            # if num_modes == 1:
            #     axes = [axes]
            
            # for i in range(num_modes):
            #     axes[i].plot( ssa_components[0, i, :], label=f'Mode {i+1}')
            #     axes[i].legend()
            
            # plt.xlabel('Temps')
            # plt.show()
                
            trend = ssa_components[0, 0]  
            seasonal = ssa_components[0, 1] 
            p=ssa_components[0, 2] 
            
            # Correction finale
            sla_series =(trend_removed-seasonal)+ model.fittedvalues
            
            if len(dates) != len(sla_series):
                continue
            
            # ✅ Moyenne glissante sur 1 an (12 mois)
            df = pd.DataFrame({'sla': sla_series }, index=pd.to_datetime(dates))
            rolling_mean_1y = df['sla'].rolling(window=window_size, min_periods=1, center=True).mean()

            mask_1992_2002 = (rolling_mean_1y.index.year >= 1992) & (rolling_mean_1y.index.year <= 2002)
            mask_2002_2012 = (rolling_mean_1y.index.year >= 2002) & (rolling_mean_1y.index.year <= 2012)
            mask_2012_2022 = (rolling_mean_1y.index.year > 2012) & (rolling_mean_1y.index.year <= 2021)

            trend_1992_2002, trend_error_1992_2002, conf_interval_1992_2002, r_squared_1992_2002 = compute_trend_gmm(
                rolling_mean_1y.index.year[mask_1992_2002], rolling_mean_1y.values[mask_1992_2002]
            )

            trend_2002_2012, trend_error_2002_2012, conf_interval_2002_2012, r_squared_2002_2012 = compute_trend_gmm(
                rolling_mean_1y.index.year[mask_2002_2012], rolling_mean_1y.values[mask_2002_2012]
            )

            trend_2012_2022, trend_error_2012_2022, conf_interval_2012_2022, r_squared_2012_2022 = compute_trend_gmm(
                rolling_mean_1y.index.year[mask_2012_2022], rolling_mean_1y.values[mask_2012_2022]
            )

            sla_time_series[station] = {
                "sla_series": rolling_mean_1y.values,
                "trend_1992_2002": trend_1992_2002,
                "trend_error_1992_2002": trend_error_1992_2002,
                "r_squared_1992_2002": r_squared_1992_2002,
                "trend_2002_2012": trend_2002_2012,
                "trend_error_2002_2012": trend_error_2002_2012,
                "r_squared_2002_2012": r_squared_2002_2012,
                "trend_2012_2022": trend_2012_2022,
                "trend_error_2012_2022": trend_error_2012_2022,
                "r_squared_2012_2022": r_squared_2012_2022,
                "dates": rolling_mean_1y.index,
                "longitude": lon_nearest,
                "latitude": lat_nearest,
                "la": lat_selected,
                "lo": lon_selected,
                "distance": dist
            }

        ds.close()
    return sla_time_series

def bezier_curve(control_points, t):
    """
    Calcule une courbe de Bézier à partir de points de contrôle.
    :param control_points: liste ou tableau des points de contrôle de la courbe de Bézier
    :param t: un tableau de paramètres t (de 0 à 1) pour calculer la courbe
    :return: valeurs interpolées sur la courbe de Bézier
    """
    n = len(control_points) - 1
    result = np.zeros(len(t))  # Initialisation du tableau de résultats
    for i in range(n + 1):
        binomial_coeff = comb(n, i)
        result += binomial_coeff * ((1 - t) ** (n - i)) * (t ** i) * control_points[i]
    return result

# Fonction d'interpolation des corrections via courbe de Bézier
def interpolate_with_bezier(dac):
    """
    Interpole les valeurs de correction avec une courbe de Bézier à partir des données non aberrantes.
    :param dac: tableau 2D des valeurs de correction avec des valeurs aberrantes supprimées (NaN)
    :return: tableau 2D de valeurs interpolées
    """
    # Créer un tableau pour stocker les résultats interpolés
    interpolated_dac = np.copy(dac)
    
    # Appliquer l'interpolation sur chaque ligne (chaque trace du satellite)
    for i in range(dac.shape[0]):  # Chaque ligne (chaque trace)
        row = dac[i, :]
        
        # Trouver les indices des valeurs non aberrantes (non NaN)
        non_nan_indices = np.where(~np.isnan(row))[0]
        non_nan_values = row[non_nan_indices]
        
        if len(non_nan_indices) > 1:  # Si nous avons plus d'une valeur non NaN
            # Créer les points de contrôle sous forme de tuples (index, valeur)
            control_points = np.array([(non_nan_indices[j], non_nan_values[j]) for j in range(len(non_nan_indices))])

            # Créer un ensemble de paramètres t (de 0 à 1) pour l'interpolation
            t = np.linspace(0, 1, len(row))  # Points t pour interpoler sur toute la ligne

            # Calculer la courbe de Bézier pour la ligne
            bezier_values = bezier_curve(control_points[:, 1], t)

            # Remplacer les NaN dans la ligne par les valeurs interpolées
            row[np.isnan(row)] = bezier_values[np.isnan(row)]
            
            # Mettre à jour la ligne dans la matrice interpolée
            interpolated_dac[i, :] = row
    
    return interpolated_dac
#----------données seanoe----------      
window_sizes_SLA = {
    "APALACHICOLA":35,
    "CEDAR KEY II": 50,
    "DAUPHIN ISLAND": 37,
    "FORT MYERS": 65,
    "FREEPORT": 50,
    "GALVESTON I": 70, 
    "GALVESTON II": 65,
    "GRAND ISLE": 50,
    "KEY WEST": 70,
    "NAPLES": 65,
    "PENSACOLA": 50,
    "PORT ISABEL": 50,
    "ROCKPORT": 50,
    "ST. PETERSBURG": 50,
    "VERACRUZ": 70,
} 
nearest_points_s = find_nearest_sla_points(seanoe,coordonnees,60)
#plot_nearest_points_on_map(nearest_points_s)

sla_time_series_s = {}
all_lat_selected = []
all_lon_selected = []
station_=[]

for file in seanoe:
    data = xr.open_dataset(file, decode_times=False)
    
    # Mise à jour des coordonnées pour chaque fichier
    lat = data['lat'].data
    lon = data['lon'].data
    
    for station, coord in nearest_points_s.items():
        lat_nearest = coord["latitude"]
        lon_nearest = coord["longitude"]

        # Vérifier que la station est dans la zone couverte par le fichier
        if not (lat.min() <= lat_nearest <= lat.max() and lon.min() <= lon_nearest <= lon.max()):
            print(f"Station {station} hors de la zone du fichier {file}, skipping...")
            continue
        
        # Trouver les indices des 10 points les plus proches avec cKDTree
        dist, idx_spatial = find_nearest_idx(lat_nearest, lon_nearest, lat, lon, k=10)
        
        if np.any(dist == np.inf):  # Vérifier si aucun point valide n'a été trouvé
            print(f"Aucun point valide trouvé pour {station} dans {file}, skipping...")
            continue
        
        if station in sla_time_series_s:
            prev_dist = sla_time_series_s[station]['distance']
            if dist >= prev_dist:
                continue
        # Extraire les indices des points valides
        best_point_numbers = data["nbpoints"].values[idx_spatial]
        sorted_idx = np.argsort(dist)  # Tri **croissant** (vers la côte)
        best_point_numbers = best_point_numbers[sorted_idx[:10]]
        valid_indices = best_point_numbers < len(data['sla'])
        best_point_numbers = best_point_numbers[valid_indices]

        if len(best_point_numbers) == 0:
            print(f"Indices invalides pour {station} dans {file}, skipping...")
            continue
 

        # Extraire et moyenner la SLA en intégrant progressivement les points les plus proches
        sla_series = data['sla'].isel(nbpoints=best_point_numbers).mean(dim="nbpoints").values
        sla_10 = data['sla_mean_10pts'].values

        # Extraire les coordonnées moyennes des points sélectionnés
        lat_selected = data['lat'].isel(nbpoints=best_point_numbers).values
        lon_selected = data['lon'].isel(nbpoints=best_point_numbers).values
        all_lat_selected.extend(lat_selected)
        all_lon_selected.extend(lon_selected)
        station_.extend([station] * len(lat_selected))

        # Si tu veux les afficher sur une carte
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        # # Affichage de la carte
        # ax.set_extent([-100, -80, 17, 32], crs=ccrs.PlateCarree())
        # ax.coastlines(resolution='50m', color='black', linewidth=1)

        # # Points originaux et voisins
        # ax.scatter(lon, lat, label="Points originaux", color="blue", alpha=0.5, transform=ccrs.PlateCarree())
        # ax.scatter(lon_selected, lat_selected, label="10 points les plus proches", color="red", marker="x", transform=ccrs.PlateCarree())
        # ax.scatter(lon_nearest, lat_nearest, label=f"Station {station}", color="green", marker="o", transform=ccrs.PlateCarree())

        # # Légende et titres
        # ax.legend()
        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')
        # ax.set_title(f"Position des points pour la station {station}")

        # plt.show()
        
        # Supprimer les NaN
        valid_indices = ~np.isnan(sla_series)
        sla_series = sla_series[valid_indices]
    

        # Convertir le temps en datetime
        tref = datetime.datetime(1950, 1, 1)
        dates_datetime = [tref + datetime.timedelta(days=int(j)) for j in data['time'].data]
        dates = [dates_datetime[i] for i in range(len(dates_datetime)) if valid_indices[i]]
        years = [date.year for date in dates]
        d=pd.DataFrame(dates)
        time_num=(d[0]-d[0].min()).dt.total_seconds().values
        
        X = sm.add_constant(time_num)
        model = sm.GLS(sla_series, X).fit()
        trend_removed = sla_series - model.fittedvalues  
        window_size_sla = window_sizes_SLA.get(station)
        ssa = SingularSpectrumAnalysis(window_size=window_size_sla , groups='auto')
        ssa_components = ssa.fit_transform(trend_removed.reshape(1, -1))
        num_modes = ssa_components.shape[1]  # Nombre de modes réellement disponibles
        
        fig, axes = plt.subplots(num_modes, 1, figsize=(10, 2 * num_modes), sharex=True)
        
        # S'assurer que axes est toujours une liste
        if num_modes == 1:
            axes = [axes]
        
        for i in range(num_modes):
            axes[i].plot( ssa_components[0, i, :], label=f'Mode {i+1}')
            axes[i].legend()
        
        plt.xlabel('Temps')
        plt.show()
            
        trend = ssa_components[0, 0]  
        seasonal = ssa_components[0, 1] 
        p=ssa_components[0, 2] 
        
        # Correction finale
        sla_series =(trend_removed-seasonal)+ model.fittedvalues
        
        if len(years) != len(sla_series):
            print(f"Les tailles ne correspondent pas pour {station} dans {file}, skipping...")
            continue

        # Séparation des périodes
        mask_2002_2012 = (np.array(years) >= 2002) & (np.array(years) <= 2012)
        mask_2012_2021 = (np.array(years) > 2012) & (np.array(years) <= 2021)

        # Calcul des tendances
        trend_2002_2012, trend_error_2002_2012, conf_interval_2002_2012,r_squared_2002_2012 = compute_trend_gmm(
            np.array(years)[mask_2002_2012], np.array(sla_series)[mask_2002_2012]
        )

        trend_2012_2021, trend_error_2012_2021, conf_interval_2012_2021,r_squared_2012_2021 = compute_trend_gmm(
            np.array(years)[mask_2012_2021], np.array(sla_series)[mask_2012_2021]
        )

        # Stockage des résultats
        sla_time_series_s[station] = {
            "sla_series": sla_series,
            "trend_2002_2012": trend_2002_2012,
            "trend_error_2002_2012": trend_error_2002_2012,
            "r_squared_2002-2012":r_squared_2002_2012,
            "trend_2012_2021": trend_2012_2021,
            "trend_error_2012_2021": trend_error_2012_2021,
            "r_squared_2012_2021":r_squared_2012_2021,
            "dates": dates,
            "longitude": lon_nearest,
            "latitude": lat_nearest,
            "distance": dist,
            "sla_mean_10pts":np.array(sla_10), 
            'temps':np.array(dates_datetime)
        }

    data.close()  # Fermer le dataset
df_export = pd.DataFrame({'Station':station_,'Latitude': all_lat_selected, 'Longitude': all_lon_selected})
# df_export.to_csv("coordonnees_pts_selectionnes_seanoe.csv", index=False, sep=";")

# #export des traces :
# longitudess=[]
# latitudess=[]
# for file in seanoe:
#     data = xr.open_dataset(file, decode_times=False)
    
#     # Extraire les coordonnées latitude et longitude à partir de la variable 'sla'
#     # Extraire les coordonnées latitude et longitude directement du Dataset
#     lat = data['lat'].values
#     lon = data['lon'].values
    
#     mask = (lon >= -100) & (lon <= -80) & (lat >= 17) & (lat <= 32)
#     lat_filtered = lat[mask]
#     lon_filtered = lon[mask]
    
#     longitudess.extend(lon_filtered)
#     latitudess.extend(lat_filtered)
    
# trace_export_s = pd.DataFrame({'Latitude': latitudess, 'Longitude':longitudess})
# trace_export_s.to_csv("Traces_seanoe.csv", index=False, sep=";")



# #---------données topex/poseidon-----------
# window_sizes_SLA_X = {
#     "APALACHICOLA":35,
#     "CEDAR KEY II": 50,
#     "DAUPHIN ISLAND": 37,
#     "FORT MYERS": 65,
#     "FREEPORT": 50,
#     "GALVESTON I": 70, 
#     "GALVESTON II": 65,
#     "GRAND ISLE": 50,
#     "KEY WEST": 70,
#     "NAPLES": 65,
#     "PENSACOLA": 50,
#     "PORT ISABEL": 50,
#     "ROCKPORT": 50,
#     "ST. PETERSBURG": 50,
#     "VERACRUZ": 70,
# } 
# nearest_points_x = find_nearest_sla_points(xtrack ,coordonnees,60)
# #plot_nearest_points_on_map(nearest_points_x)
# sla_time_series_x = {}
# seuils_satellites = {
#     "TOPEX/Poseidon": 3.0*10**-2,
#     "Jason-1": 3.3*10**-2,
#     "Jason-2": 3.4*10**-2,
#     "Jason-3": 2.0*10**-2,
# }
# for file in xtrack:
#     ds = xr.open_dataset(file, decode_times=False)
    
#     sla = ds['sla'].data
#     lat = ds['lat'].data
#     lon = ds['lon'].data
#     time = np.nanmean(ds['time'].data, axis=0)

#     dtmission = 9.9156
#     for t in range(1, len(time)):
#         if np.isnan(time[t]):
#             time[t] = time[t-1] + dtmission

#     T = np.array([tref + datetime.timedelta(days=t) for t in time])
#     id1start=min(np.where(T>tstart)[0])
#     id1end=max(np.where(T<tend)[0])
    
#     T=T[id1start:id1end]
#     ds = ds.sel(cycles_numbers=slice(id1start, id1end))
#     ds = ds.assign_coords(time=("cycles_numbers", T))

#     # id1start = np.searchsorted(T, tstart, side="left")
#     # id1end = np.searchsorted(T, tend, side="right") - 1
#     ds = ds.sel(cycles_numbers=slice(id1start, id1end))
#     lat = ds['lat'].data
#     lon = ds['lon'].data
#     #print(ds["sla"].shape) 
#     ds["sla_dac"] = ds["sla"] + ds["dac"]
#     # Appliquer tous les seuils pour chaque satellite
#     for satellite, seuil in seuils_satellites.items():
#         print(f"Fichier : {file} → Satellite : {satellite} → Seuil appliqué : {seuil}")

#         # Chargement des données
#         dac = ds['dac'].data

#         # Appliquer le seuil de 3σ ou 4σ
#         sigma_multiple = 3  # Définir ici si c'est 3σ ou 4σ en fonction de la zone
#         dac = apply_threshold_and_remove_neighbors(dac, sigma_multiple)

#         # Suppression des valeurs aberrantes basées sur le seuil satellite
#         dac[np.abs(dac) > seuil] = np.nan

#         # Mise à jour des données dans le Dataset
#         ds['dac'].data = dac
        
#     dac_interpolated = interpolate_with_bezier(dac)

#     # Remplacer les données de dac dans le dataset
#     ds['dac'].data = dac_interpolated
#     for station, coord in nearest_points_x.items():
    
#         lat_nearest = coord["latitude"]
#         lon_nearest = coord["longitude"]
        
#         # Trouver les indices des 10 points les plus proches avec cKDTree
#         dist_, idx_spatial_ = find_nearest_idx(lat_nearest, lon_nearest, lat, lon, k=10)
        
#         if np.any(dist_ == np.inf):  # Vérifier si aucun point valide n'a été trouvé
#             print(f"Aucun point valide trouvé pour {station} dans {file}, skipping...")
#             continue
        
#         # if station in sla_time_series_x:
#         #     prev_dist_ = sla_time_series_x[station]['distance_']
#         #     if dist_>= prev_dist_:
#         #         continue
#         # Extraire les indices des points valides
#         best_point_numbers = ds["points_numbers"].values[idx_spatial_]
#         sorted_idx = np.argsort(dist_)  # Tri **croissant** (vers la côte)
#         best_point_numbers = best_point_numbers[sorted_idx[:10]]
#         valid_indices = best_point_numbers < len(ds['sla'])
#         best_point_numbers = best_point_numbers[valid_indices]

#         if len(best_point_numbers) == 0:
#             print(f"Indices invalides pour {station} dans {file}, skipping...")
#             continue

#         # Extraire et moyenner la SLA en intégrant progressivement les points les plus proches
#         dac_series = ds['dac'].isel(points_numbers=best_point_numbers).mean(dim="points_numbers").values
        
#         # Trouver les indices des 10 points les plus proches avec cKDTree
#         dist, idx_spatial = find_nearest_idx(lat_nearest, lon_nearest, lat, lon, k=1)

#         if np.any(dist == np.inf):  # Vérifier si aucun point valide n'a été trouvé
#             #print(f"Aucun point valide trouvé pour {station} dans {file}, skipping...")
#             continue
        
#         if station in sla_time_series_x:
#             prev_dist = sla_time_series_x[station]['distance']
#             if dist >= prev_dist:
#                 continue
             
#         # Extraire et moyenner la SLA sur les 10 points les plus proches
#         sla_series = ds['sla'][idx_spatial, :].values
       
#         origi=ds['sla'][idx_spatial, :].values
        
#         # Extraire les coordonnées moyennes des points sélectionnés
#         lat_selected = ds['lat'].isel(points_numbers=idx_spatial).values
#         lon_selected = ds['lon'].isel(points_numbers=idx_spatial).values
        
#         valid_indices = ~np.isnan(sla_series)
#         sla_series = sla_series[valid_indices]  
        
#         dates = ds['time'].isel(cycles_numbers=slice(id1start, id1end)).values
#         dates = dates[valid_indices]
#         dates = pd.to_datetime(dates)
#         years = dates.year
        
#         #dac_series = ds['dac'][idx_spatial, :].values
#         dac_series=dac_series[valid_indices]
        
#         dac_series = pd.DataFrame({'dac':dac_series , 'dates': dates})
#         dac_series['year_month'] = pd.to_datetime(dac_series['dates']).dt.to_period('M')
#         dac_series = dac_series.groupby('year_month')['dac'].mean()
        
#         d=pd.DataFrame(dates)
#         time_num=(d[0]-d[0].min()).dt.total_seconds().values
        
#         X = sm.add_constant(time_num)
#         model = sm.GLS(sla_series, X).fit()
#         trend_removed = sla_series - model.fittedvalues  
        
#         window_size_sla_x = window_sizes_SLA_X.get(station)
#         ssa = SingularSpectrumAnalysis(window_size=window_size_sla_x , groups='auto')
#         ssa_components = ssa.fit_transform(trend_removed.reshape(1, -1))
#         # Pour reconstruire le signal complet (en utilisant toutes les composantes)
#         # reconstructed_signal = np.sum(ssa_components, axis=0)
#         # plt.plot(reconstructed_signal, label='Signal reconstruit')
#         # plt.plot(trend_removed, label='Signal original')
#         # plt.legend()
#         # plt.show()

#         num_modes = ssa_components.shape[1]  # Nombre de modes réellement disponibles
        
#         # fig, axes = plt.subplots(num_modes, 1, figsize=(10, 2 * num_modes), sharex=True)
        
#         # # S'assurer que axes est toujours une liste
#         # if num_modes == 1:
#         #     axes = [axes]
        
#         # for i in range(num_modes):
#         #     axes[i].plot( ssa_components[0, i, :], label=f'Mode {i+1}')
#         #     axes[i].legend()
        
#         # plt.xlabel('Temps')
#         # plt.show()
            
#         trend = ssa_components[0, 0]  
#         seasonal = ssa_components[0, 1] 
#         p=ssa_components[0, 2] 
        
#         # Correction finale
#         sla_series =(trend_removed-seasonal)+ model.fittedvalues

#         if len(years) != len(sla_series):
#             #print(f"Les tailles ne correspondent pas pour {station}, skipping...")
#             continue

#         df = pd.DataFrame({'sla': sla_series , 'dates': dates})
#         df['year_month'] = pd.to_datetime(df['dates']).dt.to_period('M')
#         monthly_mean = df.groupby('year_month')['sla'].mean()
        

#         mask_1992_2002 = (monthly_mean.index.year >= 1992) & (monthly_mean.index.year <= 2002)
#         mask_2002_2012 = (monthly_mean.index.year >= 2002) & (monthly_mean.index.year <= 2012)
#         mask_2012_2022 = (monthly_mean.index.year > 2012) & (monthly_mean.index.year <= 2021)

#         trend_1992_2002, trend_error_1992_2002, conf_interval_1992_2002, r_squared_1992_2002 = compute_trend_gmm(
#             monthly_mean.index.year[mask_1992_2002], monthly_mean.values[mask_1992_2002]
#         )

#         trend_2002_2012, trend_error_2002_2012, conf_interval_2002_2012, r_squared_2002_2012 = compute_trend_gmm(
#             monthly_mean.index.year[mask_2002_2012], monthly_mean.values[mask_2002_2012]
#         )

#         trend_2012_2022, trend_error_2012_2022, conf_interval_2012_2022, r_squared_2012_2022 = compute_trend_gmm(
#             monthly_mean.index.year[mask_2012_2022], monthly_mean.values[mask_2012_2022]
#         )

#         sla_time_series_x[station] = {
#             "sla_series": monthly_mean.values,
#             "sla_ori":origi,
#             "trend_1992_2002": trend_1992_2002,
#             "trend_error_1992_2002": trend_error_1992_2002,
#             "r_squared_1992_2002": r_squared_1992_2002,
#             "trend_2002_2012": trend_2002_2012,
#             "trend_error_2002_2012": trend_error_2002_2012,
#             "r_squared_2002_2012": r_squared_2002_2012,
#             "trend_2012_2022": trend_2012_2022,
#             "trend_error_2012_2022": trend_error_2012_2022,
#             "r_squared_2012_2022": r_squared_2012_2022,
#             "dates": monthly_mean.index.to_timestamp(),
#             "longitude": lon_nearest,
#             "latitude": lat_nearest,
#             "la": lat_selected,
#             "lo": lon_selected,
#             "distance": dist,
#             'distance_':dist_,
#             "tps":ds['time'],
#             "DAC":dac_series 
#         }


#     ds.close()  
    
# for station, data in sla_time_series_x.items():
#     for stat, da in sla_time_series_s.items():
#         if station == stat:
#             dates = data["dates"]
#             sla_series = data["sla_series"]
#             dates_s=da['dates']
#             sla_series_s = da["sla_series"]
        
#             plt.figure(figsize=(10, 5))
#             plt.plot(dates, sla_series, label="SLA X", color='blue', alpha=0.6)
#             plt.plot(dates_s, sla_series_s, label="SLA S", color='red', alpha=0.6)
            
#             plt.xlabel("Date")
#             plt.ylabel("SLA (m)")
#             plt.title(f"Série temporelle SLA - Station {station}")
#             plt.legend()
#             plt.grid()
#             plt.show()
            
# for station in sla_time_series_x : 
#     # Si les coordonnées sont valides, tracer la carte
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
#     # Affichage de la carte
#     ax.set_extent([-100, -80, 17, 32], crs=ccrs.PlateCarree())
#     ax.coastlines(resolution='50m', color='black', linewidth=1)
    
#     # Points originaux et voisins
#     ax.scatter(sla_time_series_x[station]['longitude'], sla_time_series_x[station]['latitude'], label="Points originaux", color="blue", alpha=0.5, transform=ccrs.PlateCarree())
#     ax.scatter(sla_time_series_x[station]['lo'], sla_time_series_x[station]['la'], label="Point sélectionné", color="red", marker="x", transform=ccrs.PlateCarree())

    
#     # Légende et titres
#     ax.legend()
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_title(f"Position des points pour la station {station}")
    
#     plt.show()  

#export des points les plus proches :
# data_export = []  # Liste pour stocker les données avant l'export
# for station, values  in sla_time_series_x.items():
#     la = values["la"]
#     lo = values["lo"]
#     data_export.append({"Station": station, "Latitude": la, "Longitude": lo})


# df_export_x = pd.DataFrame(data_export)
# df_export_x.to_csv("coordonnees_pts_selectionnes_TP.csv", index=False, sep=";")

#export des traces :
# longitudes=[]
# latitudes=[]
# for file in xtrack:
#     data = xr.open_dataset(file, decode_times=False)
    
#     # Extraire les coordonnées latitude et longitude à partir de la variable 'sla'
#     # Extraire les coordonnées latitude et longitude directement du Dataset
#     lat = data['lat'].values
#     lon = data['lon'].values

#     mask = (lon >= -100) & (lon <= -80) & (lat >= 17) & (lat <= 32)
#     lat_filtered = lat[mask]
#     lon_filtered = lon[mask]
    
#     longitudes.extend(lon_filtered)
#     latitudes.extend(lat_filtered)
    
# trace_export_x = pd.DataFrame({'Latitude': latitudes, 'Longitude':longitudes})
# trace_export_x.to_csv("Traces_TP.csv", index=False, sep=";")

#      #---------moyenne glissante de 1 an---------
# window_sizes_SLA_X_1 = {
#     "APALACHICOLA":35,
#     "CEDAR KEY II": 50,
#     "DAUPHIN ISLAND": 37,
#     "FORT MYERS": 65,
#     "FREEPORT": 50,
#     "GALVESTON I": 70, 
#     "GALVESTON II": 65,
#     "GRAND ISLE": 11,
#     "KEY WEST": 70,
#     "NAPLES": 65,
#     "PENSACOLA": 50,
#     "PORT ISABEL": 50,
#     "ROCKPORT": 50,
#     "ST. PETERSBURG": 50,
#     "VERACRUZ": 70,
# } 
# sla_time_series_x_1y = compute_sla_with_rolling_mean_x(xtrack, nearest_points_x,12,window_sizes_SLA_X_1)
# for station, data in sla_time_series_x_1y.items():
#     for stat, da in sla_time_series_s.items():
#         if station == stat:
#             if station == "GRAND ISLE":
#                 dates = data["dates"]
#                 sla_series = data["sla_series"]
#                 dates_s=da['dates']
#                 sla_series_s = da["sla_series"]
            
#                 plt.figure(figsize=(10, 5))
#                 plt.plot(dates, sla_series, label="SLA X", color='blue', alpha=0.6)
#                 plt.plot(dates_s, sla_series_s, label="SLA S", color='red', alpha=0.6)
                
#                 plt.xlabel("Date")
#                 plt.ylabel("SLA (m)")
#                 plt.title(f"Série temporelle SLA - Station {station}")
#                 plt.legend()
#                 plt.grid()
#                 plt.show()
            
# for station in sla_time_series_x_1y : 
#     # Si les coordonnées sont valides, tracer la carte
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
#     # Affichage de la carte
#     ax.set_extent([-100, -80, 17, 32], crs=ccrs.PlateCarree())
#     ax.coastlines(resolution='50m', color='black', linewidth=1)
    
#     # Points originaux et voisins
#     ax.scatter(sla_time_series_x_1y[station]['longitude'], sla_time_series_x_1y[station]['latitude'], label="Points originaux", color="blue", alpha=0.5, transform=ccrs.PlateCarree())
#     ax.scatter(sla_time_series_x_1y[station]['lo'], sla_time_series_x_1y[station]['la'], label="Point sélectionné", color="red", marker="x", transform=ccrs.PlateCarree())

    
#     # Légende et titres
#     ax.legend()
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_title(f"Position des points pour la station {station}")
    
#     plt.show()
     #---------moyenne glissante de 3 an---------
# window_sizes_SLA_X_3 = {
#     "APALACHICOLA":5,
#     "CEDAR KEY II": 5,
#     "DAUPHIN ISLAND": 5,
#     "FORT MYERS": 5,
#     "FREEPORT": 5,
#     "GALVESTON I": 5, 
#     "GALVESTON II": 5,
#     "GRAND ISLE": 4,
#     "KEY WEST": 5,
#     "NAPLES": 5,
#     "PENSACOLA": 2,
#     "PORT ISABEL": 2,
#     "ROCKPORT": 5,
#     "ST. PETERSBURG": 5,
#     "VERACRUZ": 5,
# } 
# sla_time_series_x_3y = compute_sla_with_rolling_mean_x(xtrack, nearest_points_x,36,window_sizes_SLA_X_3)
#for station, data in sla_time_series_x_3y.items():
#     for stat, da in sla_time_series_s.items():
#         if station == stat:
#             dates = data["dates"]
#             sla_series = data["sla_series"]
#             dates_s=da['dates']
#             sla_series_s = da["sla_series"]
        
#             plt.figure(figsize=(10, 5))
#             plt.plot(dates, sla_series, label="SLA X", color='blue', alpha=0.6)
#             plt.plot(dates_s, sla_series_s, label="SLA S", color='red', alpha=0.6)
            
#             plt.xlabel("Date")
#             plt.ylabel("SLA (m)")
#             plt.title(f"Série temporelle SLA - Station {station}")
#             plt.legend()
#             plt.grid()
#             plt.show()
                
# for station in sla_time_series_x_3y : 
#     # Si les coordonnées sont valides, tracer la carte
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
#     # Affichage de la carte
#     ax.set_extent([-100, -80, 17, 32], crs=ccrs.PlateCarree())
#     ax.coastlines(resolution='50m', color='black', linewidth=1)
    
#     # Points originaux et voisins
#     ax.scatter(sla_time_series_x_3y[station]['longitude'], sla_time_series_x_3y[station]['latitude'], label="Points originaux", color="blue", alpha=0.5, transform=ccrs.PlateCarree())
#     ax.scatter(sla_time_series_x_3y[station]['lo'], sla_time_series_x_3y[station]['la'], label="Point sélectionné", color="red", marker="x", transform=ccrs.PlateCarree())

    
#     # Légende et titres
#     ax.legend()
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_title(f"Position des points pour la station {station}")
    
#     plt.show()
#sla_time_series_x_2y = compute_sla_with_rolling_mean_x(xtrack, nearest_points_x,24)
#export des points les plus proches :
# data_export = []  # Liste pour stocker les données avant l'export
# for station, values  in sla_time_series_x_3y.items():
#     la = values["la"]
#     lo = values["lo"]
#     data_export.append({"Station": station, "Latitude": la, "Longitude": lo})


# df_export_x = pd.DataFrame(data_export)
# df_export_x.to_csv("coordonnees_pts_selectionnes_TP.csv", index=False, sep=";")

#%% DAC SEANOE 

# Chemin vers les fichiers .nc
path_xtrack = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\GOM_XTRACK_LP2\*.nc"

# Utilisation de glob pour obtenir tous les fichiers .nc
files = glob.glob(path_xtrack)

numeros_recherches = ["154","041","117","193","230","052","015","128","024","091","026","102","167"]  # Liste des numéros à ouvrir

# Filtrer les fichiers qui contiennent l'un des numéros recherchés
matching_files = [f for f in files if any(num in f for num in numeros_recherches)]

DAC_Xtrack ={}

for file in matching_files:
    ds = xr.open_dataset(file, decode_times=False)
    
    sla = ds['sla'].data
    lat = ds['lat'].data
    lon = ds['lon'].data
    time = np.nanmean(ds['time'].data, axis=0)

    dtmission = 9.9156
    for t in range(1, len(time)):
        if np.isnan(time[t]):
            time[t] = time[t-1] + dtmission

    T = np.array([tref + datetime.timedelta(days=t) for t in time])
    id1start=min(np.where(T>tstart)[0])
    id1end=max(np.where(T<tend)[0])
    
    T=T[id1start:id1end]
    ds = ds.sel(cycles_numbers=slice(id1start, id1end))
    ds = ds.assign_coords(time=("cycles_numbers", T))

    # id1start = np.searchsorted(T, tstart, side="left")
    # id1end = np.searchsorted(T, tend, side="right") - 1
    ds = ds.sel(cycles_numbers=slice(id1start, id1end))
    lat = ds['lat'].data
    lon = ds['lon'].data
    #print(ds["sla"].shape) 
    ds["sla_dac"] = ds["sla"] + ds["dac"]
    # Appliquer tous les seuils pour chaque satellite
    for satellite, seuil in seuils_satellites.items():

        # Chargement des données
        dac = ds['dac'].data

        # Appliquer le seuil de 3σ ou 4σ
        sigma_multiple = 3  # Définir ici si c'est 3σ ou 4σ en fonction de la zone
        dac = apply_threshold_and_remove_neighbors(dac, sigma_multiple)

        # Suppression des valeurs aberrantes basées sur le seuil satellite
        dac[np.abs(dac) > seuil] = np.nan

        # Mise à jour des données dans le Dataset
        ds['dac'].data = dac
        
    dac_interpolated = interpolate_with_bezier(dac)

    # Remplacer les données de dac dans le dataset
    ds['dac'].data = dac_interpolated
    name = os.path.basename(file)
    DAC_Xtrack[name] = {
        "tps": ds['time'].data,
        "DAC": ds['dac'].data,
        "lat":lat,
        "lon":lon
    }
DAC_mensuel_extremites = {}

for name, contenu in DAC_Xtrack.items():
    tps = contenu["tps"]
    dac = contenu["DAC"]  # shape: [n_points_alongtrack, n_cycles]
    lat = contenu["lat"]
    lon = contenu["lon"]

    # Conversion du temps en datetime
    dates = pd.to_datetime(tps)

    # DataFrame : index = temps, colonnes = points (le long de la trace)
    df = pd.DataFrame(dac.T, index=dates, columns=np.arange(dac.shape[0]))

    # Moyenne mensuelle
    df_mensuel = df.resample("M").mean()
    mask = (df_mensuel.index >= "2002-01-01") & (df_mensuel.index <= "2021-12-31")
    df_mensuel = df_mensuel.loc[mask]
    # Moyenne des 10 premiers points (extrémité 1)
    moyenne_ext1 = df_mensuel.iloc[:, :10].mean(axis=1)
    lat_ext1 = np.nanmean(lat[:10])
    lon_ext1 = np.nanmean(lon[:10])

    # Moyenne des 10 derniers points (extrémité 2)
    moyenne_ext2 = df_mensuel.iloc[:, -10:].mean(axis=1)
    lat_ext2 = np.nanmean(lat[-10:])
    lon_ext2 = np.nanmean(lon[-10:])

    DAC_mensuel_extremites[name] = {
        "extremite_1": {
            "dac": moyenne_ext1,
            "lat": lat_ext1,
            "lon": lon_ext1
        },
        "extremite_2": {
            "dac": moyenne_ext2,
            "lat": lat_ext2,
            "lon": lon_ext2
        }
    }
#---S---
txt_file_path = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\liste fichier seanoe.txt"

with open(txt_file_path, 'r') as f:
    filenames = f.read().splitlines()

files = [f"C:\\Users\\camil\\OneDrive\\Bureau\\Univ_de_la_Rochelle\\M2\\semestre 10\\SEANEO\\{filename}" for filename in filenames]

seasea ={}

for file in files:
    ds = xr.open_dataset(file)
    
    temps = ds['time'].data
    print(temps)
    lat = ds['lat'].data
    lon = ds['lon'].data
    sla=ds['sla_mean_10pts'].data
    name = os.path.basename(file)
    seasea[name] = {
        "tps": temps,
        "lat":lat,
        "lon":lon,
        'sla':sla
    }

from geopy.distance import geodesic

for name, contenu in seasea.items():
    lat_s = contenu["lat"]
    lon_s = contenu["lon"]
    sla = contenu["sla"]
    tps = pd.to_datetime(contenu["tps"])

    # Initialiser la meilleure distance
    best_dist = np.inf
    best_dac = None
    best_source = None  # Pour enregistrer le nom de la trace et l'extrémité

    lat_s_val = float(lat_s[0]) if isinstance(lat_s, (np.ndarray, list)) else float(lat_s)
    lon_s_val = float(lon_s[0]) if isinstance(lon_s, (np.ndarray, list)) else float(lon_s)

    for nom_trace, extremites in DAC_mensuel_extremites.items():
        for ext in ["extremite_1", "extremite_2"]:
            lat_x = extremites[ext]["lat"]
            lon_x = extremites[ext]["lon"]
            dist = geodesic((lat_s_val, lon_s_val), (lat_x, lon_x)).km

            if dist < best_dist:
                best_dist = dist
                best_dac = extremites[ext]["dac"]
                best_source = f"{nom_trace} - {ext}"

    # Interpolation de DAC aux dates SEANOE
    dac_interp = best_dac.reindex(tps, method='nearest')

    # Ajouter DAC à SLA
    sla_corrige = sla + dac_interp.values

    # Mise à jour du dictionnaire
    seasea[name]["dac"] = dac_interp.values
    seasea[name]["sla_corrige"] = sla_corrige
    seasea[name]["dac_source"] = best_source  # ← ici tu ajoutes la trace utilisée

# Dictionnaire pour stocker les séries corrigées
dict_sla_corrige = {}

for i, (name, contenu) in enumerate(seasea.items()):
    tps = pd.to_datetime(contenu["tps"])
    sla_corrige = contenu["sla_corrige"]
    dict_sla_corrige[i] = pd.Series(sla_corrige, index=tps)

# Combine tout en un DataFrame
df_sla_corrige = pd.DataFrame(dict_sla_corrige)

#%%

gdf = gpd.read_file(r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\courbes de niveaux .gpkg")
if 'ELEV' in gdf.columns:
    gdf400 = gdf[gdf['ELEV'] <= -400]  # Filtrer les courbes de niveau à -200 m ou moins
else:
    print("La colonne 'elevation' n'a pas été trouvée dans le fichier .gpkg")
if 'ELEV' in gdf.columns:
    gdf200 = gdf[gdf['ELEV'] == -200]
# gdf200.plot(ax=ax, edgecolor='red', linewidth=0.9)        
# gdf400.plot(ax=ax, edgecolor='black', linewidth=0.5)

# Initialisation de la carte
m = folium.Map(location=[25, -90], zoom_start=6)

gdf = gpd.read_file(r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\courbes de niveaux .gpkg")

# Vérifier si la colonne 'ELEV' existe
if 'ELEV' in gdf.columns:
    gdf400 = gdf[gdf['ELEV'] <= -400]  # Filtrer les courbes ≤ -400 m
    gdf200 = gdf[gdf['ELEV'] == -200]  # Filtrer la courbe à -200 m
    
    # Ajouter les courbes de niveau à la carte
    folium.GeoJson(
        gdf400,
        name="Courbes ≤ -400m",
        style_function=lambda x: {"color": "black", "weight": 0.5},
    ).add_to(m)

    folium.GeoJson(
        gdf200,
        name="Courbe -200m",
        style_function=lambda x: {"color": "red", "weight": 0.9},
    ).add_to(m)
    
# Création des groupes de fonctionnalités
trace_tp = folium.FeatureGroup(name="Trace altimétrique T/P").add_to(m)
trace_seanoe = folium.FeatureGroup(name="Trace altimétrique Seanoe").add_to(m)
pt_seanoe = folium.FeatureGroup(name="Point le plus proche pour Seanoe").add_to(m)
station_layer = folium.FeatureGroup(name="Marégraphes").add_to(m)

station_colors = {
    'APALACHICOLA': 'red',
    'CEDAR KEY II': 'blue',
    'DAUPHIN ISLAND': 'chartreuse',
    'FORT MYERS': 'salmon',
    'FREEPORT': 'burlywood',
    'GALVESTON I': 'pink',
    'GALVESTON II': 'yellow',
    'GRAND ISLE': 'brown',
    'KEY WEST': 'black',
    'NAPLES': 'white',
    'PENSACOLA': 'cyan',
    'PORT ISABEL': 'magenta',
    'ROCKPORT': 'olive',
    'ST. PETERSBURG': 'indigo',
    'VERACRUZ': 'grey',
    
}

# Ajouter les marqueurs pour Seanoe
for file in seanoe:
    data = xr.open_dataset(file, decode_times=False)
    match = re.search(r'_(\d{3}_\d{2})-', file)
    trace_number = match.group(1) if match else "N/A"

    latitudes = data['lat'].data
    longitudes = data['lon'].data
    latitudes_s = np.where(data['lon'].data)[0]
    distance=(data['distance_to_coast'].data)*10**-3
    
    # Vérifie si les coordonnées sont en tableau ou scalaires
    for lat, lon,idx ,d in zip(latitudes, longitudes, latitudes_s,distance):
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color='green',
            popup=f"<b>Indice:</b> {idx}<br><b>Distance:</b> {d} Km<br>",
            tooltip=f"Trace: {trace_number}",
            fill=True,
            fill_color='green',
            fill_opacity=0.7,
        ).add_to(trace_seanoe)


# Ajouter les marqueurs pour T/P
for file in xtrack:
    ds = xr.open_dataset(file, decode_times=False)
    trace_number = re.search(r'(\d{3})\.nc$', os.path.basename(file))
    trace_number = trace_number.group(0) if trace_number else "Unknown"
    

    sla = ds['sla'].data
    latitudes = ds['lat'].data
    longitudes = ds['lon'].data
    time = np.nanmean(ds['time'].data, axis=0)
    latitudes_ = np.where(ds['lon'].data)[0]
    dtmission = 9.9156
    for t in range(1, len(time)):
        if np.isnan(time[t]):
            time[t] = time[t-1] + dtmission

    T = np.array([tref + datetime.timedelta(days=t) for t in time])
    ds = ds.assign_coords(time=("cycles_numbers", T))

    # Sélection des données entre tstart et tend
    id1start = np.searchsorted(T, tstart, side="left")
    id1end = np.searchsorted(T, tend, side="right") - 1
    ds = ds.sel(cycles_numbers=slice(id1start, id1end))
    
    latitudes = ds['lat'].data
    longitudes = ds['lon'].data
    
    # Ajout des marqueurs
    for lat, lon,ids in zip(latitudes, longitudes,latitudes_):
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color='orange',
            tooltip=f"Trace {trace_number}",
            popup=ids,
            fill=True,
            fill_color='orange',
            fill_opacity=0.7,
        ).add_to(trace_tp)

# Ajouter les stations marégraphiques dans un cluster
marker_cluster = MarkerCluster().add_to(station_layer)

for Station, data in dataframes.items():
    if Station in station_colors:
        color = station_colors[Station]
    else:
        color = 'gray' 
    latitude = data["lat"].unique()[0]  # Utiliser le premier élément si c'est une liste
    longitude = data["lon"].unique()[0]
    
    # Récupérer les dates de début et de fin à partir du DataFrame
    start_date = data["temps"].min().strftime("%Y-%m-%d")
    end_date = data["temps"].max().strftime("%Y-%m-%d")
    temps_couverture = (data["temps"].max() - data["temps"].min()).days
    nb_annees = round(temps_couverture / 365.25, 0)
    
    # Calculer les lacunes
    nb_valeurs = len(data)
    nb_gaps = (data["hauteur d'eau en mm"] == -99999).sum()  # Compter les -99999 comme gaps
    pourcentage_gaps = round((nb_gaps / nb_valeurs) * 100, 1)
    
    # Ajouter un marqueur avec un tooltip et un popup
    folium.RegularPolygonMarker(
        location=[latitude, longitude],
        number_of_sides= 5,
        radius=8,
        color='black',
        weight=0.7,
        tooltip=Station,  # Le nom de la station
        popup=f"<b>Début:</b> {start_date}<br><b>Fin:</b> {end_date}<br><b>Temps de couverture:</b> {nb_annees}<br><b>Lacunes (%):</b> {pourcentage_gaps}",
        fill=True,
        fill_color=color,
        fill_opacity=1,
    ).add_to(station_layer)

for _, row in df_export.iterrows():  # Itérer correctement sur les lignes du DataFrame
    station = row['Station']
    
    if station in station_colors:
        color = station_colors[station]
    else:
        color = 'gray' 
    
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],  # Accéder correctement aux valeurs
        radius=4,
        color='black',
        weight=0.7,
        fill=True,
        fill_color=color,
        fill_opacity=1,
    ).add_to(pt_seanoe)


# Ajouter le contrôle des couches
folium.LayerControl().add_to(m)

# Sauvegarder la carte en HTML
m.save("Carte traces seanoe + Xtrack +pts_seanoe+pts_xtrack.html")

#%%-----------NaN contenu dans les séries sla X-TRACK---------------
# nan_data = []
# for station in sla_time_series_s.keys():
#     data_x = sla_time_series_x.get(station, None)  # Vérifier si la station est aussi dans X

#     # Si la station existe aussi dans sla_time_series_x, ajouter ses données
#     if data_x:
#      sla_x = np.array(data_x["sla_series"])  # Conversion en mm
#      ori=np.array(data_x["sla_ori"])  
#      tps=pd.to_datetime(data_x["tps"])
#      years=np.array(tps.year)
#      dates_x = pd.to_datetime(data_x["dates"])
#      years_x = np.array(dates_x.year)
#      # Définition des périodes
#      periods = [(1992, 2002), (2002, 2012), (2012, 2022)]
        
#     # Comptage des NaN pour chaque période
#      for start, end in periods:
#          mask = (years >= start) & (years < end)  # Filtrer les années
#          nan_count = np.isnan(ori[mask]).sum()   # Compter les NaN
#          nan_data.append({"Période": f"{start}-{end}", "Station": station, "NaN": nan_count})

         
#          # Création de la figure
#      plt.figure(figsize=(20, 8))
    
#      plt.plot(tps,ori)
#      # plt.scatter(dates_x, sla_x, label=f"SLA {station} (série X)", color='green', s=10)
#      # plt.plot(dates_x, sla_x, label=f"SLA {station} (série X)", color='green')
#      plt.title(f"Série temporelle SLA pour {station}", fontsize=14)
#      plt.xlabel('Date', fontsize=12)
#      plt.ylabel('SLA (mm)', fontsize=12)
#      plt.legend()
#      plt.grid(True)
#      plt.xticks(rotation=45)
#      plt.tight_layout()

#      plt.show()
    
# # Création du DataFrame
# df_nan = pd.DataFrame(nan_data)
#%%---------------Correction TG du DAC de X-TRACK-------------
# station_results = {}

# # Boucle sur les clés du DataFrame
# for key, df in dataframes.items():
#     # Traiter la première série SLA
#     for station, values in sla_time_series_x.items():
#         if station == key:
#             mask_tg = (df['temps'].dt.year >= 2002) & (df['temps'].dt.year <= 2021)
#             df_filtered = df.loc[mask_tg].copy().dropna()
            
#             # Convertir le temps en secondes depuis la première date
#             time_numeric = (df_filtered ['temps'] - df_filtered ['temps'].min()).dt.total_seconds().values
             
#             # Créer et rééchantillonner les données SLA
#             dac_dates = pd.to_datetime(values['dates'])
            
#             dac_df = pd.DataFrame({'date': dac_dates, 'DAC': values['DAC']})
            
#             dac_df.set_index('date', inplace=True)
#             #dac_df= dac_df.resample('M').mean()
            
#             # Filtrer entre 2002 et 2021
#             mask_dac = (dac_df.index.year >= 2002) & (dac_df.index.year <= 2021)
#             dac_filtered = dac_df.loc[mask_dac].dropna()
            
#             # Aligner les données SLA
#             aligned_dac = dac_filtered.reindex(df_filtered['temps'], method='nearest')
#             df_filtered['DAC'] = aligned_dac.values
            
#             if len(df_filtered) == len(aligned_dac):
#                 df_filtered['TG_corriger'] = (df_filtered['hauteur d\'eau en mm'].values)*10**-3 - df_filtered['DAC']
#             else:
#                 print(f"Taille incompatible pour {station} : TG ({len(df_filtered)}) vs DAC ({len(dac_filtered)})")
                
#             # print(df_filtered['temps'].diff().value_counts())  # Vérifie l'intervalle entre les points
#             X = sm.add_constant(time_numeric)  # Ajouter une constante pour l'intercept
#             model = sm.OLS(df_filtered['TG_corriger'], X).fit()  # Ajuster le modèle GLS
#             trend_removed = df_filtered['TG_corriger'] - model.fittedvalues  
            
#             # Transformer la série en signal fréquentiel
  
#             # y = trend_removed.dropna().values
#             # n = len(y)
#             # freqs = np.fft.fftfreq(n, d=1)  # d=1 si les données sont régulières
#             # fft_vals = np.abs(fft(y))
            
#             # # Tracer le spectre
#             # plt.figure(figsize=(10, 5))
#             # plt.plot(1 / freqs[1:n//2], fft_vals[1:n//2])  # Convertir en période (1/fréquence)
#             # plt.xlabel("Période")
#             # plt.ylabel("Amplitude")
#             # plt.title("Spectre de fréquence (FFT)")
#             # plt.show()
#             # # Trouver les pics dominants
#             # indices = np.argsort(fft_vals)[-5:][::-1]  # Les 5 plus grandes amplitudes
#             # dominant_periods = 1 / freqs[indices]
            
#             # # Filtrer les valeurs infinies ou négatives
#             # dominant_periods = [round(p, 2) for p in dominant_periods if p > 0 and np.isfinite(p)]
            
#             # print("Périodes dominantes détectées :", dominant_periods)
            
#             mstl = MSTL(trend_removed , periods=[6,12,24])  # Ajuster les périodes

#             decomposed = mstl.fit()
#             # decomposed .plot()
#             # plt.tight_layout()
#             # plt.show()
#             seasonal = decomposed.seasonal 
#             trend=decomposed.trend
#             # # Étape 3: Vérification après suppression de la saisonnalité
#             deseasoned =trend_removed-seasonal['seasonal_24']-seasonal['seasonal_12']-seasonal['seasonal_6']

#             # # Réintroduire la tendance initiale
#             adjusted = deseasoned # Réintroduire la tendance initiale

#             # Mettre à jour le DataFrame
#             df_filtered ['TG_corriger'] = adjusted
            
#             # Appliquer la moyenne glissante
#             #df_filtered['TG_corriger'] = df_filtered['TG_corriger'] .rolling(window=6, min_periods=1).mean()

#             # Mise à jour du DataFrame
#     dataframes[key] = df_filtered

   
# for key, df in dataframes.items():
#     # Créer une figure par station
#     plt.figure(figsize=(20, 8))
#     # Pour le deuxième ensemble de données SLA
#     for station, values in sla_time_series_s.items():
#         if station == key:
#             dac_dates = pd.to_datetime(values['dates'])
#             sla_df = pd.DataFrame({'date': dac_dates, 'sla': values['sla_series']})
            
#             time_numeric = (sla_df ['date'] - sla_df['date'].min()).dt.total_seconds().values
            
#             X = sm.add_constant(time_numeric)  # Ajouter une constante pour l'intercept
#             model_gls = sm.OLS(sla_df['sla'], X).fit()  # Ajuster le modèle GLS
#             trend_removed = sla_df['sla'] - model_gls.fittedvalues  
            
#             mstl = MSTL(trend_removed,periods=[6,12])
            
#             decomposed = mstl.fit()
#             # decomposed.plot()
#             # plt.tight_layout()
#             # plt.show()
#             seasonal = decomposed.seasonal 
            
#             deseasoned = trend_removed - seasonal['seasonal_6']-seasonal['seasonal_12']  

#             adjusted = deseasoned # Réintroduire la tendance initiale
#             sla_df['sla'] = adjusted
#             sla_df['sla']= sla_df['sla'].rolling(window=6, min_periods=1).mean()
#             sla_df.set_index('date', inplace=True)
            
#             # Rééchantillonner les données SLA
#             aligned_sla = sla_df.reindex(df['temps'], method='nearest')
#             aligned_sla_values = aligned_sla['sla'].values
#             #aligned_sla_values = pd.Series(aligned_sla_values).rolling(window=12, min_periods=6).mean().values
            
#             # Calcul de VLM
#             VLM = df['TG_corriger']-aligned_sla_values
#             df['VLM'] = VLM
#             dataframes[key] = df
            
#             # Régression linéaire et polynomial, puis calcul de R²
          
#             times_numeric = (df['temps'] - df['temps'].min()).dt.days.values
        
#             # # Ajustement linéaire et polynomial
#             # p_coeff_linear = np.polyfit(times_numeric, (df['TG_corriger']).values, 1)
#             # p_poly_linear = np.poly1d(p_coeff_linear)
#             # p_coeff_poly = np.polyfit(times_numeric, (df['TG_corriger']).values, 2)
#             # p_poly = np.poly1d(p_coeff_poly)
            
#             # # Calcul des valeurs prédites et R²
#             # predicted_values_linear = p_poly_linear(times_numeric)
#             # predicted_values_poly = p_poly(times_numeric)
#             # r2_linear = r2_score((df['TG_corriger']).values, predicted_values_linear)
#             # r2_poly = r2_score((df['TG_corriger']).values, predicted_values_poly)
            
#             # Extraire les dates et SLA
#             tps = pd.to_datetime(aligned_sla.index)
#             sla = aligned_sla_values 
#             if key == "GRAND ISLE":
#                 output_df = pd.DataFrame({'date': tps, 'sla': sla})
#                 output_df.to_csv("GRAND_ISLE_SLA.txt", sep="\t", index=False, header=True)
#             # corr_coef, p_value = pearsonr((df['TG_corriger']).values, sla)
        
    
#             # Tracer les résultats
#             plt.plot(tps,sla*10**3 , label=f"SLA ({station})", color='red')
#             plt.plot(df['temps'], (df['TG_corriger'].values)*10**3, label=f"VLM ajusté ({station})", color='blue')
        
                
#             plt.title(f"Série temporelle pour {station} - {key}", fontsize=14)
#             plt.xlabel('Date', fontsize=12)
#             plt.ylabel('Hauteur (mm)', fontsize=12)
#             plt.legend()
#             plt.grid(True)
#             plt.ylim(-100, 100)
#             plt.yticks(range(-100, 101, 50))
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             plt.show()
#             Ajouter les résultats pour cette station dans le dictionnaire
#             station_results[station] = {
#                 "R² linéaire": r2_linear,
#                 "R² polynomial": r2_poly,
#                 "Corrélation de Pearson": corr_coef,
#                 "P-value": p_value
#             }

# # Affichage des résultats pour toutes les stations après le traitement
# for station, results in station_results.items():
#     print(f"Résultats pour {station}:")
#     print(f"  R² linéaire : {results['R² linéaire']:.2f}")
#     print(f"  R² polynomial (degré 2) : {results['R² polynomial']:.2f}")
#     print(f"  Corrélation de Pearson : {results['Corrélation de Pearson']:.2f}")
#     print(f"  P-value : {results['P-value']:.2f}")

#%% comparer tg et seanoe avec SSA 

# station_results = {}
# save_directory_SSA_TG_SEA = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\Comparaison Seanoe et TG après SSA"
# window_sizes_TG = {
#     "APALACHICOLA": 30,
#     "CEDAR KEY II": 50,
#     "DAUPHIN ISLAND":38,
#     "FORT MYERS": 40,
#     "FREEPORT": 50,
#     "GALVESTON I": 70, 
#     "GALVESTON II": 30,
#     "GRAND ISLE": 25,
#     "KEY WEST": 70,
#     "NAPLES": 28,
#     "PENSACOLA": 45,
#     "PORT ISABEL": 50,
#     "ROCKPORT": 30,
#     "ST. PETERSBURG":40,
#     "VERACRUZ": 70,
# }

# window_sizes_SLA = {
#     "APALACHICOLA":35,
#     "CEDAR KEY II": 50,
#     "DAUPHIN ISLAND": 37,
#     "FORT MYERS": 65,
#     "FREEPORT": 50,
#     "GALVESTON I": 70, 
#     "GALVESTON II": 65,
#     "GRAND ISLE": 50,
#     "KEY WEST": 70,
#     "NAPLES": 65,
#     "PENSACOLA": 50,
#     "PORT ISABEL": 50,
#     "ROCKPORT": 50,
#     "ST. PETERSBURG": 50,
#     "VERACRUZ": 70,
# }

# # Boucle sur les clés du DataFrame
# for key, df in dataframes.items():
#     #plt.figure(figsize=(20, 8))
#     for station, values in sla_time_series_x.items():
#         if station == key:
#             mask_tg = (df['temps'].dt.year >= 2002) & (df['temps'].dt.year <= 2021)
#             df_filtered = df.loc[mask_tg].copy().dropna()
#             # Convertir le temps en secondes depuis la première date
#             time_numeric = (df_filtered['temps'] - df_filtered['temps'].min()).dt.total_seconds().values

#             # Créer et rééchantillonner les données SLA
#             dac_dates = pd.to_datetime(values['dates'])
#             dac_df = pd.DataFrame({'date': dac_dates, 'DAC': values['DAC']})
#             dac_df.set_index('date', inplace=True)

#             # Filtrer entre 2002 et 2021
#             mask_dac = (dac_df.index.year >= 2002) & (dac_df.index.year <= 2021)
#             dac_filtered = dac_df.loc[mask_dac].dropna()

#             # Aligner les données SLA
#             aligned_dac = dac_filtered.reindex(df_filtered['temps'], method='nearest')
#             df_filtered['DAC'] = aligned_dac.values

#             if len(df_filtered) == len(aligned_dac):
#                 df_filtered['TG_corriger'] = (df_filtered["hauteur d'eau en mm"].values) * 10**-3 - df_filtered['DAC'].values
#             else:
#                 print(f"Taille incompatible pour {station} : TG ({len(df_filtered)}) vs DAC ({len(dac_filtered)})")
            
#             # Ajustement OLS pour la tendance
#             X = sm.add_constant(time_numeric)
#             model = sm.GLS(df_filtered['TG_corriger'], X).fit()
#             trend_removed = df_filtered['TG_corriger'] - model.fittedvalues  
#             window_size_tg = window_sizes_TG.get(station) 

#             # SSA pour extraire la tendance et la saisonnalité
#             ssa = SingularSpectrumAnalysis(window_size=window_size_tg , groups='auto')
#             ssa_components = ssa.fit_transform(trend_removed.values.reshape(1, -1))
#             num_modes = ssa_components.shape[1]  # Nombre de modes réellement disponibles
#             # fig, axes = plt.subplots(num_modes, 1, figsize=(10, 2 * num_modes), sharex=True)
            
#             # # S'assurer que axes est toujours une liste
#             # if num_modes == 1:
#             #     axes = [axes]
            
#             # for i in range(num_modes):
#             #     axes[i].plot( ssa_components[0, i, :], label=f'Mode {i+1}')
#             #     axes[i].legend()
            
#             # plt.xlabel('Temps')
#             # plt.show()

#             trend = ssa_components[0, 0]  
#             seasonal = ssa_components[0, 1]
#             p=ssa_components[0, 2] 
            
#             # Correction finale
#             df_filtered["TG_corriger"] =trend_removed-seasonal

#             # Appliquer une moyenne glissante
#             df_filtered['TG_corriger'] = df_filtered['TG_corriger'].rolling(window=6, min_periods=1).mean()

#             # Mise à jour du DataFrame
#             dataframes[key] = df_filtered

# # Deuxième boucle pour le SLA et le calcul du VLM
# for key, df in dataframes.items():
#     plt.figure(figsize=(20, 8))

#     for station, values in sla_time_series_s.items():
#         if station == key:
#             dac_dates = pd.to_datetime(values['temps'])
#             sla_mean_10=values['sla_mean_10pts']
#             sla_df = pd.DataFrame({'date': dac_dates, 'sla': sla_mean_10})
#             sla_df=sla_df.dropna()
#             time_numeric = (sla_df['date'] - sla_df['date'].min()).dt.total_seconds().values

#             # Ajustement OLS
#             X = sm.add_constant(time_numeric)
#             model_gls = sm.GLS(sla_df['sla'], X).fit()
#             trend_removed = sla_df['sla'] - model_gls.fittedvalues  

#             # SSA pour SLA
#             window_size_sla = window_sizes_SLA.get(station) 
#             ssa_sla = SingularSpectrumAnalysis(window_size=window_size_sla, groups='auto')
#             ssa_components_sla = ssa_sla.fit_transform(trend_removed.values.reshape(1, -1))
#             # for i in range(ssa_components_sla.shape[1]):
#             #     plt.subplot(ssa_components_sla.shape[1], 1, i + 1)
#             #     plt.plot(ssa_components_sla[0, i])
#             #     plt.title(f"Mode SSA {i+1}")
#             #     plt.xlabel('Temps')
#             #     plt.ylabel('Valeur')
#             #     plt.legend()
#             trend_sla = ssa_components_sla[0, 0]
#             seasonal_sla = ssa_components_sla[0, 1]
#             r=ssa_components_sla[0, 2]

#             adjusted_sla = trend_removed-seasonal_sla
#             sla_df['sla'] = adjusted_sla
#             sla_df['sla'] = sla_df['sla'].rolling(window=6, min_periods=1).mean()
#             sla_df.set_index('date', inplace=True)

#             # Rééchantillonnage SLA sur TG
#             aligned_sla = sla_df.reindex(df['temps'], method='nearest')
#             aligned_sla_values = aligned_sla['sla'].values

#             # Calcul du VLM
#             df['VLM'] = df['TG_corriger'] - aligned_sla_values
#             dataframes[key] = df
            
           
#             # Tracer les résultats
#             plt.plot(pd.to_datetime(aligned_sla.index), aligned_sla_values * 10**3, label="SLA", color='red')
#             plt.plot(df['temps'], df['TG_corriger'].values * 10**3, label="Marégraphe", color='blue')

#             plt.title(f"Série temporelle pour {station}", fontsize=14)
#             plt.xlabel('Année', fontsize=14)
#             plt.ylabel('Hauteur (mm)', fontsize=14)
#             plt.legend()
#             plt.grid(True)
#             plt.ylim(-100, 100)
#             plt.yticks(range(-100, 101, 50))
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             #plt.xticks(pd.date_range(start=df['temps'].min(), end=df['temps'].max(), freq='YS'))
#             #plt.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
#             plt.savefig(os.path.join(save_directory_SSA_TG_SEA, f"{station}.png"), dpi=400)
#             plt.show()

#%% comparer seanoe, xtrack et TG avec SSA -> PENSACOLA
# dtmission = 9.9156
# t_26=xr.open_dataset(r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\GOM_XTRACK_LP2\ctoh.sla.ref.TP+J1+J2+J3.gom.026.nc",
#                      decode_times=False)

# time_26 = np.nanmean(t_26['time'].data, axis=0)

# for t in range(1, len(time_26)):
#     if np.isnan(time_26[t]):
#         time_26[t] = time_26[t-1] + dtmission

# T_26 = np.array([tref + datetime.timedelta(days=t) for t in time_26])
# id1start_26=min(np.where(T_26>tstart)[0])
# id1end_26=max(np.where(T_26<tend)[0])

# T_26=T_26[id1start_26:id1end_26]
# t_26 = t_26.sel(cycles_numbers=slice(id1start_26, id1end_26))
# t_26 = t_26.assign_coords(time=("cycles_numbers", T_26))
# t_26 = t_26.sel(cycles_numbers=slice(id1start_26, id1end_26))

# idx_26 = np.where(t_26['lat'].data)[0]

# t_15=xr.open_dataset(r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\GOM_XTRACK_LP2\ctoh.sla.ref.TP+J1+J2+J3.gom.015.nc",
#                      decode_times=False)

# time_15 = np.nanmean(t_15 ['time'].data, axis=0)

# for t in range(1, len(time_15 )):
#     if np.isnan(time_15 [t]):
#         time_15 [t] = time_15 [t-1] + dtmission

# T_15  = np.array([tref + datetime.timedelta(days=t) for t in time_15 ])
# id1start_15 =min(np.where(T_15 >tstart)[0])
# id1end_15 =max(np.where(T_15 <tend)[0])

# T_15 =T_15 [id1start_15 :id1end_15 ]
# t_15  = t_15.sel(cycles_numbers=slice(id1start_15, id1end_15))
# t_15  = t_15 .assign_coords(time=("cycles_numbers", T_15 ))
# t_15  = t_15 .sel(cycles_numbers=slice(id1start_15, id1end_15))
# lat=t_15['lat'].data
# lon=t_15['lon'].data


# idx_15= np.where(t_15['lat'].data)[0]

# Tps_15=t_15['time'].data
# Tps_26=t_26['time'].data

# # Indices de référence
# ref_idx_15 = 224
# ref_idx_26 = 339

# # Définir un rayon pour sélectionner les indices voisins autour des points de référence
# radius = 3  # Par exemple, on veut 5 points avant et après le point de référence

# # Sélectionner les indices voisins autour des indices de référence
# idx_15_nearby = list(range(max(0, ref_idx_15 - radius), min(len(t_15['sla'].data), ref_idx_15 + radius + 1)))
# idx_26_nearby = list(range(max(0, ref_idx_26 - radius), min(len(t_26['sla'].data), ref_idx_26 + radius + 1)))

# # Créer les DataFrames pour les points voisins
# sla_15_nearby = pd.DataFrame([t_15['sla'].data[i] for i in idx_15_nearby]).T
# sla_26_nearby = pd.DataFrame([t_26['sla'].data[i] for i in idx_26_nearby]).T

# sla_15_nearby['temps'] = pd.to_datetime(Tps_15)
# sla_26_nearby['temps'] = pd.to_datetime(Tps_26)


# # Arrondir les dates à la journée (en supprimant les heures, minutes et secondes)
# sla_15_nearby['date'] = pd.to_datetime(Tps_15).date
# sla_26_nearby['date'] = pd.to_datetime(Tps_26).date

# # Fusionner les DataFrames autour de la date
# df_aligned = pd.merge(sla_15_nearby[['date', 0,1,2,3,4,5,6]], sla_26_nearby[['date', 0,1,2,3,4,5,6]], on='date', suffixes=('_15', '_26'))
# df_aligned.set_index('date',inplace=True)

# # Calculer la matrice de corrélation de Pearson
# correlation_matrix = df_aligned.corr()

# print("Matrice de corrélation entre les points voisins :")
# print(correlation_matrix)

# # Si tu veux calculer la corrélation entre tous les points de sla_15 et sla_26 voisins
# correlation_pearson = correlation_matrix['0_15'].iloc[1:]  # Exclure la corrélation de la diagonale
# print("Corrélation de Pearson pour chaque point :")
# print(correlation_pearson)

# # Matrice de corrélation (supposons que df_aligned existe déjà)
# correlation_matrix = df_aligned.corr(method='pearson')

# # # Création de la carte thermique (heatmap)
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)

# # # Titre du graphique
# # plt.title('Matrice de Corrélation entre SLA_15 et SLA_26')

# # # Afficher le graphique
# # plt.show()

# sla_values = []
# dates = []

# for idx, row in sla_15_nearby.iterrows():
#     for col in sla_15_nearby.columns[0:6]: 
#         sla_values.append(row[col])
#         dates.append(row['temps'])

# for idx, row in sla_26_nearby.iterrows():
#     for col in sla_26_nearby.columns[0:6]: 
#         sla_values.append(row[col])
#         dates.append(row['temps'])


# sla_15_26 = pd.DataFrame({'date': dates, 'sla': sla_values})
# sla_15_26 = sla_15_26.sort_values(by='date').reset_index(drop=True)
# sla_15_26=sla_15_26.dropna()
# mask= (sla_15_26['date'].dt.year >= 2002) & (sla_15_26['date'].dt.year <= 2021)
# sla_15_26= sla_15_26.loc[mask]
# sla_15_26 = sla_15_26.set_index('date')
# sla_15_26 = sla_15_26.resample('M').mean()

# window_size=50

# # SSA pour extraire la tendance et la saisonnalité
# ssa = SingularSpectrumAnalysis(window_size=window_size , groups='auto')
# ssa_components = ssa.fit_transform(sla_15_26.values.reshape(1, -1))
# n_modes = ssa_components.shape[1]  # Nombre de modes extraits
# time_index_x= sla_15_26.index  # Garder l'axe du temps correct
# num_modes = ssa_components.shape[1]  # Nombre de modes réellement disponibles
# fs = 1  # Fréquence d'échantillonnage (1 point par mois)

# # Initialisation des groupes de modes
# low_freq_modes = []  # Modes de basse fréquence (> 2 ans)
# high_freq_modes = []  # Modes < 24 mois

# # Analyse spectrale pour classer les modes
# for mode in range(num_modes):
#     signal = ssa_components[0, mode]
#     N = len(signal)

#     # FFT
#     freq = np.fft.fftfreq(N, d=1/fs)
#     fft_values = np.fft.fft(signal) 
#     power_spectrum = np.abs(fft_values)**2

#     # Récupérer les périodes associées aux fréquences positives
#     positive_freqs = freq[freq > 0]
#     positive_spectrum = power_spectrum[freq > 0]
#     periods = 1 / positive_freqs  # Conversion en périodes (mois)

#     # Identifier la période dominante
#     peak_period = periods[np.argmax(positive_spectrum)]

#     # Classification des modes
#     if peak_period >= 24:  # Conserver les modes de basse fréquence (> 2 ans)
#         low_freq_modes.append(mode)
        
#     else:  # Modes à retirer (saisonnalité et bruit haute fréquence)
#         high_freq_modes.append(mode)
# hig_xt=np.sum(ssa_components[0, high_freq_modes],axis=0)
# # fig, axes = plt.subplots(n_modes, 1, figsize=(10, 2 * n_modes), sharex=True)

# # # S'assurer que axes est toujours une liste
# # if n_modes == 1:
# #     axes = [axes]

# # for i in range(n_modes):
# #     axes[i].plot(time_index, ssa_components[0, i, :], label=f'Mode {i+1}')
# #     axes[i].legend()

# # plt.xlabel('Temps')
# # plt.show()

# # # Correction finale
# # sla_15_26['sla'] =sla_15_26['sla'] -hig_xt
# # plt.figure(figsize=(20, 8))
# # #---mode 1---
# mode_1_x=ssa_components[0, 0]
# moy_mode_1_x=np.mean(mode_1_x)
# std_mode_1_x=np.std(mode_1_x)
# mode_stand_1_x = (mode_1_x - moy_mode_1_x)/std_mode_1_x

# # plt.subplot(3,1,1)
# # plt.plot(time_index,ssa_components[0, 0],label="Mode 1")
# # plt.plot(time_index,mode_stand_1_x,label="Mode 1 standardisé" )
# # plt.legend()

# # #---mode 2---
# mode_2_x=ssa_components[0, 1]
# moy_mode_2_x=np.mean(mode_2_x)
# std_mode_2_x=np.std(mode_2_x)
# mode_stand_2_x = (mode_2_x - moy_mode_2_x)/std_mode_2_x

# # plt.subplot(3,1,2)
# # plt.plot(time_index,ssa_components[0, 1],label="Mode 2")
# # plt.plot(time_index,mode_stand_2_x,label="Mode 2 standardisé")
# # plt.legend()

# # #---mode 3---
# mode_3x=ssa_components[0, 2]
# moy_mode_3x=np.mean(mode_3x)
# std_mode_3x=np.std(mode_3x)
# mode_stand_3x = (mode_3x - moy_mode_3x)/std_mode_3x

# # plt.subplot(3,1,3)
# # plt.plot(time_index,ssa_components[0, 2],label="Mode 3")
# # plt.plot(time_index,mode_stand_3x,label="Mode 3 standardisé")
# # plt.legend()
# # plt.show()
 

# #----------Seanoe----------
# s=xr.open_dataset(r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\SEANEO\ESACCI-SEALEVEL-IND-MSLTR-MERGED-R4_JA_015_01-20241122-v2.4.nc",
#                         decode_times=False)
# s_15_1=pd.DataFrame(s['sla_mean_10pts'].data,columns=['sla'])
# s_15_1['temps']=[tref + datetime.timedelta(days=int(j)) for j in s['time'].data]
# s_15_1=s_15_1.dropna()

# s_15_1= s_15_1.set_index('temps')

# time_index_s = s_15_1.index 

# ssa_sla = SingularSpectrumAnalysis(window_size=50, groups='auto')
# ssa_components_sla = ssa_sla.fit_transform(s_15_1.values.reshape(1, -1))
# num_modes = ssa_components_sla .shape[1]  # Nombre de modes réellement disponibles
# fs = 1  # Fréquence d'échantillonnage (1 point par mois)

# # Initialisation des groupes de modes
# low_freq_modes = []  # Modes de basse fréquence (> 2 ans)
# high_freq_modes = []  # Modes < 24 mois

# # Analyse spectrale pour classer les modes
# for mode in range(num_modes):
#     signal = ssa_components[0, mode]
#     N = len(signal)

#     # FFT
#     freq = np.fft.fftfreq(N, d=1/fs)
#     fft_values = np.fft.fft(signal) 
#     power_spectrum = np.abs(fft_values)**2

#     # Récupérer les périodes associées aux fréquences positives
#     positive_freqs = freq[freq > 0]
#     positive_spectrum = power_spectrum[freq > 0]
#     periods = 1 / positive_freqs  # Conversion en périodes (mois)

#     # Identifier la période dominante
#     peak_period = periods[np.argmax(positive_spectrum)]

#     # Classification des modes
#     if peak_period >= 24:  # Conserver les modes de basse fréquence (> 2 ans)
#         low_freq_modes.append(mode)
        
#     else:  # Modes à retirer (saisonnalité et bruit haute fréquence)
#         high_freq_modes.append(mode)
# hig_sla =np.sum(ssa_components_sla [0, high_freq_modes],axis=0)
# # fig, axes = plt.subplots(n_modes, 1, figsize=(10, 2 * n_modes), sharex=True)

# # # S'assurer que axes est toujours une liste
# # if n_modes == 1:
# #     axes = [axes]

# # for i in range(n_modes):
# #     axes[i].plot(time_index, ssa_components_sla[0, i, :], label=f'Mode {i+1}')
# #     axes[i].legend()

# # plt.xlabel('Temps')
# # plt.show()

# # s_15_1['sla'] =s_15_1['sla']-hig_sla

# # # Rééchantillonnage SLA sur TG
# # s_15_1= s_15_1.reindex(sla_15_26.index, method='nearest')    
# # plt.figure(figsize=(20, 8))
# # #---mode 1---
# mode_1s=ssa_components_sla[0, 0]
# moy_mode_1s=np.mean(mode_1s)
# std_mode_1s=np.std(mode_1s)
# mode_stand_1s = (mode_1s - moy_mode_1s)/std_mode_1s

# # plt.subplot(3,1,1)
# # plt.plot(time_index,ssa_components_sla[0, 0],label="Mode 1")
# # plt.plot(time_index,mode_stand_1s,label="Mode 1 standardisé" )
# # plt.legend()

# # #---mode 2---
# mode_2s=ssa_components_sla[0, 1]
# moy_mode_2s=np.mean(mode_2s)
# std_mode_2s=np.std(mode_2s)
# mode_stand_2s = (mode_2s - moy_mode_2s)/std_mode_2s

# # plt.subplot(3,1,2)
# # plt.plot(time_index,ssa_components_sla[0, 1],label="Mode 2")
# # plt.plot(time_index,mode_stand_2s,label="Mode 2 standardisé")
# # plt.legend()

# # #---mode 3---
# mode_3s=ssa_components_sla[0, 2]
# moy_mode_3s=np.mean(mode_3s)
# std_mode_3s=np.std(mode_3s)
# mode_stand_3s = (mode_3s - moy_mode_3s)/std_mode_3s

# # plt.subplot(3,1,3)
# # plt.plot(time_index,ssa_components_sla[0, 2],label="Mode 3")
# # plt.plot(time_index,mode_stand_3s,label="Mode 3 standardisé")
# # plt.legend()
# # plt.show()
 
# #-----------TG-----------
# # Boucle sur les clés du DataFrame
# for key, df in dataframes.items():
#     #plt.figure(figsize=(20, 8))
#     for station, values in sla_time_series_x.items():
#         if station == key:
#             if station == "PENSACOLA" and key=="PENSACOLA":
#                 mask_tg = (df['temps'].dt.year >= 2002) & (df['temps'].dt.year <= 2021)
#                 df_filtered = df.loc[mask_tg].copy().dropna()
#                 #df_filtered = df.copy().dropna()

    
#                 # Créer et rééchantillonner les données SLA
#                 dac_dates = pd.to_datetime(values['dates'])
#                 dac_df = pd.DataFrame({'date': dac_dates, 'DAC': values['DAC']})
#                 dac_df.set_index('date', inplace=True)
    
#                 # Filtrer entre 2002 et 2021
                
#                 dac_filtered = dac_df.dropna()
    
#                 # Aligner les données SLA
#                 aligned_dac = dac_filtered.reindex(df_filtered['temps'], method='nearest')
#                 df_filtered['DAC'] = aligned_dac.values
    
#                 if len(df_filtered) == len(aligned_dac):
#                     df_filtered['TG_corriger'] = ((df_filtered["hauteur d'eau en mm"].values) * 10**-3 )- df_filtered['DAC'].values
#                 else:
#                     print(f"Taille incompatible pour {station} : TG ({len(df_filtered)}) vs DAC ({len(dac_filtered)})")

#                 # SSA pour extraire la tendance et la saisonnalité
#                 ssa = SingularSpectrumAnalysis(window_size=50 , groups='auto')
#                 ssa_components = ssa.fit_transform( df_filtered['TG_corriger'].values.reshape(1, -1))
#                 num_modes = ssa_components.shape[1]  # Nombre de modes réellement disponibles
#                 fs = 1  # Fréquence d'échantillonnage (1 point par mois)
                
#                 # Initialisation des groupes de modes
#                 low_freq_modes = []  # Modes de basse fréquence (> 2 ans)
#                 high_freq_modes = []  # Modes < 24 mois
                
#                 # Analyse spectrale pour classer les modes
#                 for mode in range(num_modes):
#                     signal = ssa_components[0, mode]
#                     N = len(signal)
                
#                     # FFT
#                     freq = np.fft.fftfreq(N, d=1/fs)
#                     fft_values = np.fft.fft(signal) 
#                     power_spectrum = np.abs(fft_values)**2
                
#                     # Récupérer les périodes associées aux fréquences positives
#                     positive_freqs = freq[freq > 0]
#                     positive_spectrum = power_spectrum[freq > 0]
#                     periods = 1 / positive_freqs  # Conversion en périodes (mois)
                
#                     # Identifier la période dominante
#                     peak_period = periods[np.argmax(positive_spectrum)]
                
#                     # Classification des modes
#                     if peak_period >= 24:  # Conserver les modes de basse fréquence (> 2 ans)
#                         low_freq_modes.append(mode)
                        
#                     else:  # Modes à retirer (saisonnalité et bruit haute fréquence)
#                         high_freq_modes.append(mode)
                        
#                 hig=np.sum(ssa_components[0, high_freq_modes],axis=0)
#                 # plt.figure(figsize=(20, 8))
#                 # plt.plot(np.sum(ssa_components[0, low_freq_modes],axis=0))
#                 # plt.show()
#                 # fig, axes = plt.subplots(num_modes, 1, figsize=(10, 2 * num_modes), sharex=True)
                
#                 # # S'assurer que axes est toujours une liste
#                 # if num_modes == 1:
#                 #     axes = [axes]
                
#                 # for i in range(num_modes):
#                 #     axes[i].plot(df_filtered['temps'], ssa_components[0, i, :], label=f'Mode {i+1}')
#                 #     axes[i].legend()
                
#                 # plt.xlabel('Temps')
#                 # plt.show()
             
#                 df_filtered['TG_corriger']=df_filtered['TG_corriger']-hig
#                 # Mise à jour du DataFrame
#                 dataframes[key] = df_filtered
                                
#                 mode_1=ssa_components[0, 0]
#                 moy_mode_1=np.mean(mode_1)
#                 std_mode_1=np.std(mode_1)
#                 mode_stand_1 = (mode_1 - moy_mode_1)/std_mode_1
                
#                 #---mode 2---
#                 mode_2=ssa_components[0, 1]
#                 moy_mode_2=np.mean(mode_2)
#                 std_mode_2=np.std(mode_2)
#                 mode_stand_2 = (mode_2 - moy_mode_2)/std_mode_2
#                 #---mode 3---
#                 mode_3=ssa_components[0, 2]
#                 moy_mode_3=np.mean(mode_3)
#                 std_mode_3=np.std(mode_3)
#                 mode_stand_3 = (mode_3 - moy_mode_3)/std_mode_3
                
# plt.figure(figsize=(20, 8))
# plt.subplot(3,1,1)
# plt.plot(df_filtered['temps'],mode_stand_1,label="Mode 1 TG" ,color="blue")
# plt.plot(time_index_x,mode_stand_1_x,label="Mode 1 xtrack",color="green" )
# plt.plot(time_index_s,mode_stand_1s,label="Mode 1 seanoe",color="red" )
# plt.legend()


# plt.subplot(3,1,2)
# plt.plot(df_filtered['temps'],mode_stand_2,label="Mode 2 TG",color="blue")
# plt.plot(time_index_x,mode_stand_2_x,label="Mode 2 xtrack",color="green")
# plt.plot(time_index_s,mode_stand_2s,label="Mode 2 seanoe",color="red")
# plt.legend()

# plt.subplot(3,1,3)
# plt.plot(df_filtered['temps'],mode_stand_3,label="Mode 3 TG",color="blue")
# plt.plot(time_index_x,mode_stand_3x,label="Mode 3 xtrack",color="green")
# plt.plot(time_index_s,mode_stand_3s,label="Mode 3 seanoe",color="red")
# plt.legend()
# plt.show()

#%%
def open_d(i, dataset):    
    s = pd.DataFrame(dataset[i]['sla_mean_10pts'].data, columns=['sla'])
    t=dataset[i]['time'].data
    s['temps'] = [tref + datetime.timedelta(days=int(j)) for j in t]
    s = s.dropna()
    
    s = s.set_index('temps')
    time_index_s = s.index 
    
    ssa_sla = SingularSpectrumAnalysis(window_size=50, groups='auto')
    ssa_components_sla = ssa_sla.fit_transform(s.values.reshape(1, -1))
    num_modes = ssa_components_sla.shape[1]  # Nombre de modes réellement disponibles
    fs = 1  # Fréquence d'échantillonnage (1 point par mois)
    
    # Initialisation des groupes de modes
    low_freq_modes = []  # Modes de basse fréquence (> 2 ans)
    high_freq_modes = []  # Modes < 24 mois
    
    # Analyse spectrale pour classer les modes
    for mode in range(num_modes):
        signal = ssa_components_sla[0, mode]
        N = len(signal)
    
        # FFT
        freq = np.fft.fftfreq(N, d=1/fs)
        fft_values = np.fft.fft(signal) 
        power_spectrum = np.abs(fft_values)**2
    
        # Récupérer les périodes associées aux fréquences positives
        positive_freqs = freq[freq > 0]
        positive_spectrum = power_spectrum[freq > 0]
        periods = 1 / positive_freqs  # Conversion en périodes (mois)
    
        # Identifier la période dominante
        peak_period = periods[np.argmax(positive_spectrum)]
    
        # Classification des modes
        if peak_period >= 24:  # Conserver les modes de basse fréquence (> 2 ans)
            low_freq_modes.append(mode)
        else:  # Modes à retirer (saisonnalité et bruit haute fréquence)
            high_freq_modes.append(mode)
    
    hig_sla = np.sum(ssa_components_sla[0, high_freq_modes], axis=0)

    # ---mode 1---
    mode_1s = ssa_components_sla[0, 0]
    moy_mode_1s = np.mean(mode_1s)
    std_mode_1s = np.std(mode_1s)
    mode_stand_1s = (mode_1s - moy_mode_1s) / std_mode_1s
    
    # ---mode 2---
    mode_2s = ssa_components_sla[0, 1]
    moy_mode_2s = np.mean(mode_2s)
    std_mode_2s = np.std(mode_2s)
    mode_stand_2s = (mode_2s - moy_mode_2s) / std_mode_2s
    
    # ---mode 3---
    mode_3s = ssa_components_sla[0, 2]
    moy_mode_3s = np.mean(mode_3s)
    std_mode_3s = np.std(mode_3s)
    mode_stand_3s = (mode_3s - moy_mode_3s) / std_mode_3s
    
    return time_index_s,mode_stand_1s, mode_stand_2s, mode_stand_3s, s 

# Chemin vers les fichiers .nc
path = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\SEANEO\*.nc"

# Utilisation de glob pour obtenir tous les fichiers .nc
files = glob.glob(path)

numeros_recherches = ["R4_JA_102_01", "R4_JA_102_02", "R4_JA_102_10","R4_JA_102_11"]  # Liste des numéros à ouvrir

# Filtrer les fichiers qui contiennent l'un des numéros recherchés
matching_files = [f for f in files if any(num in f for num in numeros_recherches)]

# Ouvrir les fichiers correspondants
datasets = [xr.open_dataset(f, decode_times=False) for f in matching_files]

# Appel de la fonction open_d
for i, dataset in enumerate(datasets):
    time_index_s,mode_stand_1s, mode_stand_2s, mode_stand_3s,s_CK_1 = open_d(3, datasets)

#-----------TG-----------
# Boucle sur les clés du DataFrame
for key, df in dataframes.items():
    #plt.figure(figsize=(20, 8))
    for station, values in sla_time_series_x.items():
        if station == key:
            if station == "NAPLES" and key=="NAPLES":
                # mask_tg = (df['temps'].dt.year >= 2002) & (df['temps'].dt.year <= 2021)
                # df_filtered = df.loc[mask_tg].copy().dropna()
                df_filtered = df.copy().dropna()

    
                # Créer et rééchantillonner les données SLA
                dac_dates = pd.to_datetime(values['dates'])
                dac_df = pd.DataFrame({'date': dac_dates, 'DAC': values['DAC']})
                dac_df.set_index('date', inplace=True)
    
                # Filtrer entre 2002 et 2021
                
                dac_filtered = dac_df.dropna()
    
                # Aligner les données SLA
                aligned_dac = dac_filtered.reindex(df_filtered['temps'], method='nearest')
                df_filtered['DAC'] = aligned_dac.values
    
                if len(df_filtered) == len(aligned_dac):
                    df_filtered['TG_corriger'] = ((df_filtered["hauteur d'eau en mm"].values) * 10**-3 )- df_filtered['DAC'].values
                else:
                    print(f"Taille incompatible pour {station} : TG ({len(df_filtered)}) vs DAC ({len(dac_filtered)})")

                # SSA pour extraire la tendance et la saisonnalité
                ssa = SingularSpectrumAnalysis(window_size=50 , groups='auto')
                ssa_components = ssa.fit_transform( df_filtered['TG_corriger'].values.reshape(1, -1))
                num_modes = ssa_components.shape[1]  # Nombre de modes réellement disponibles
                fs = 1  # Fréquence d'échantillonnage (1 point par mois)
                
                # Initialisation des groupes de modes
                low_freq_modes = []  # Modes de basse fréquence (> 2 ans)
                high_freq_modes = []  # Modes < 24 mois
                
                # Analyse spectrale pour classer les modes
                for mode in range(num_modes):
                    signal = ssa_components[0, mode]
                    N = len(signal)
                
                    # FFT
                    freq = np.fft.fftfreq(N, d=1/fs)
                    fft_values = np.fft.fft(signal) 
                    power_spectrum = np.abs(fft_values)**2
                
                    # Récupérer les périodes associées aux fréquences positives
                    positive_freqs = freq[freq > 0]
                    positive_spectrum = power_spectrum[freq > 0]
                    periods = 1 / positive_freqs  # Conversion en périodes (mois)
                
                    # Identifier la période dominante
                    peak_period = periods[np.argmax(positive_spectrum)]
                
                    # Classification des modes
                    if peak_period >= 24:  # Conserver les modes de basse fréquence (> 2 ans)
                        low_freq_modes.append(mode)
                        
                    else:  # Modes à retirer (saisonnalité et bruit haute fréquence)
                        high_freq_modes.append(mode)
                        
                hig=np.sum(ssa_components[0, high_freq_modes],axis=0)
             
                df_filtered['TG_corriger']=df_filtered['TG_corriger']
                # Mise à jour du DataFrame
                dataframes[key] = df_filtered
                                
                mode_1=ssa_components[0, 0]
                moy_mode_1=np.mean(mode_1)
                std_mode_1=np.std(mode_1)
                mode_stand_1 = (mode_1 - moy_mode_1)/std_mode_1
                
                #---mode 2---
                mode_2=ssa_components[0, 1]
                moy_mode_2=np.mean(mode_2)
                std_mode_2=np.std(mode_2)
                mode_stand_2 = (mode_2 - moy_mode_2)/std_mode_2
                #---mode 3---
                mode_3=ssa_components[0, 2]
                moy_mode_3=np.mean(mode_3)
                std_mode_3=np.std(mode_3)
                mode_stand_3 = (mode_3 - moy_mode_3)/std_mode_3
                
plt.figure(figsize=(20, 8))
plt.subplot(3,1,1)
plt.plot(df_filtered['temps'],mode_stand_1,label="Mode 1 TG" ,color="blue")
plt.plot(time_index_s,mode_stand_1s,label="Mode 1 seanoe",color="red" )
plt.legend()


plt.subplot(3,1,2)
plt.plot(df_filtered['temps'],mode_stand_2,label="Mode 2 TG",color="blue")
plt.plot(time_index_s,mode_stand_2s,label="Mode 2 seanoe",color="red")
plt.legend()

plt.subplot(3,1,3)
plt.plot(df_filtered['temps'],mode_stand_3,label="Mode 3 TG",color="blue")
plt.plot(time_index_s,mode_stand_3s,label="Mode 3 seanoe",color="red")
plt.legend()
plt.show()
#%% regarder si le signal est le même le long d'une trace 
# t_15=xr.open_dataset(r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\GOM_XTRACK_LP2\ctoh.sla.ref.TP+J1+J2+J3.gom.015.nc",
#                       decode_times=False)

# time_15 = np.nanmean(t_15 ['time'].data, axis=0)

# for t in range(1, len(time_15 )):
#     if np.isnan(time_15 [t]):
#         time_15 [t] = time_15 [t-1] + dtmission

# T_15  = np.array([tref + datetime.timedelta(days=t) for t in time_15 ])
# id1start_15 =min(np.where(T_15 >tstart)[0])
# id1end_15 =max(np.where(T_15 <tend)[0])

# T_15 =T_15 [id1start_15 :id1end_15 ]
# t_15  = t_15.sel(cycles_numbers=slice(id1start_15, id1end_15))
# t_15  = t_15 .assign_coords(time=("cycles_numbers", T_15 ))
# t_15  = t_15 .sel(cycles_numbers=slice(id1start_15, id1end_15))
# lat=t_15['lat'].data
# lon=t_15['lon'].data


# idx_15= np.where(t_15['lat'].data)[0]

# Tps_15=t_15['time'].data


# # Indices de référence
# ref_idx_15 =10

# # Nombre total de points à sélectionner après l'indice de référence
# num_points = 1000  # Par exemple, prendre 10 points après

# # Sélectionner les indices après le point de référence avec un pas de 5
# idx_15_nearby = list(range(ref_idx_15, min(len(t_15['sla'].data), ref_idx_15 + num_points * 5), 5))
# # Ajouter les indices 91 et 92 s'ils ne sont pas déjà présents
# if 91 not in idx_15_nearby:
#     idx_15_nearby.append(91)
# if 92 not in idx_15_nearby:
#     idx_15_nearby.append(92)

# # Créer les DataFrames pour les points sélectionnés
# sla_15_nearby = pd.DataFrame([t_15['sla'].data[i] for i in idx_15_nearby]).T
# sla_15_nearby['temps']=Tps_15
# sla_15_nearby.set_index('temps',inplace=True)
# sla_15_nearby = sla_15_nearby.resample('M').mean()
# sla_15_nearby=sla_15_nearby.dropna()
# sla_15_nearby_y = sla_15_nearby.resample('Y').mean()
# idx_15_nearby.sort()
# def ssa (sla,window_size):
#     time_numeric = (sla.index - sla.index .min()).total_seconds().values
    
#     X = sm.add_constant(time_numeric)
#     model_gls = sm.GLS(sla, X).fit()
#     trend_removed = sla - model_gls.fittedvalues  
#     time_index = trend_removed.index 

#     ssa_sla = SingularSpectrumAnalysis(window_size=window_size, groups=None)
#     ssa_components_sla = ssa_sla.fit_transform(trend_removed.values.reshape(1, -1))
    
#     # fig, axes = plt.subplots(n_modes, 1, figsize=(10, 2 * n_modes), sharex=True)
    
#     # # S'assurer que axes est toujours une liste
#     # if n_modes == 1:
#     #     axes = [axes]
    
#     # for i in range(n_modes):
#     #     axes[i].plot(time_index, ssa_components_sla[0, i, :], label=f'Mode {i+1}')
#     #     axes[i].legend()
    
#     # plt.xlabel('Temps')
#     # plt.show()
#     fs = 1  # Fréquence d'échantillonnage (1 point par mois)
#     num_modes = ssa_components_sla.shape[1]  # Nombre total de modes
    
#     # Initialisation des groupes de modes
#     low_freq_modes_10y = []  # Tendance non linéaire (> 10 ans)
#     low_freq_modes_5y = []   # Modes de 5 à 10 ans
#     low_freq_modes_2y = []   # Modes de 2 à 5 ans
#     seasonal_modes_6m = []   # Modes saisonniers de 6 mois
#     seasonal_modes_12m = []  # Modes saisonniers de 12 mois
#     other_seasonal_modes = []  # Autres modes saisonniers
#     noise_modes = []  # Bruit (haute fréquence)
    
#     # Analyse spectrale pour classer les modes
#     for mode in range(num_modes):
#         signal = ssa_components_sla[0, mode]
#         N = len(signal)
    
#         # FFT
#         freq = np.fft.fftfreq(N, d=1/fs)
#         fft_values = np.fft.fft(signal)
#         power_spectrum = np.abs(fft_values)**2
    
#         # Récupérer les périodes associées aux fréquences positives
#         positive_freqs = freq[freq > 0]
#         positive_spectrum = power_spectrum[freq > 0]
#         periods = 1 / positive_freqs  # Conversion en périodes (mois)
    
#         # Identifier la période dominante
#         peak_period = periods[np.argmax(positive_spectrum)]
    
#         # Classification des modes
#         if peak_period > 120:  # Tendance non linéaire (> 10 ans)
#             low_freq_modes_10y.append(mode)
#         elif 60 <= peak_period < 120:  # Modes de 5 à 10 ans
#             low_freq_modes_5y.append(mode)
#         elif 24 <= peak_period < 60:  # Modes de 2 à 5 ans
#             low_freq_modes_2y.append(mode)
#         elif 5 <= peak_period <= 7:  # Modes de ~6 mois
#             seasonal_modes_6m.append(mode)
#         elif 10 <= peak_period <= 14:  # Modes de ~12 mois
#             seasonal_modes_12m.append(mode)
#         elif 5 <= peak_period <= 14:  # Autres modes saisonniers
#             other_seasonal_modes.append(mode)
#         else:  # Le reste est considéré comme du bruit
#             noise_modes.append(mode)
    
#     # Vérification des modes trouvés
#     print(f"Modes >10 ans : {low_freq_modes_10y}")
#     print(f"Modes 5-10 ans : {low_freq_modes_5y}")
#     print(f"Modes 2-5 ans : {low_freq_modes_2y}")
#     print(f"Modes de 6 mois : {seasonal_modes_6m}")
#     print(f"Modes de 12 mois : {seasonal_modes_12m}")
#     print(f"Autres modes saisonniers : {other_seasonal_modes}")
#     print(f"Modes de bruit : {noise_modes}")
    
#     # Reconstruction des composantes
#     basse_freq_10 = np.sum(ssa_components_sla[0, low_freq_modes_10y], axis=0) if low_freq_modes_10y else np.zeros_like(ssa_components_sla[0, 0])
#     basse_freq_5_10 = np.sum(ssa_components_sla[0, low_freq_modes_5y], axis=0) if low_freq_modes_5y else np.zeros_like(ssa_components_sla[0, 0])
#     basse_freq_2_5 = np.sum(ssa_components_sla[0, low_freq_modes_2y], axis=0) if low_freq_modes_2y else np.zeros_like(ssa_components_sla[0, 0])
#     saisonnalite_6 = np.sum(ssa_components_sla[0, seasonal_modes_6m], axis=0) if seasonal_modes_6m else np.zeros_like(ssa_components_sla[0, 0])
#     saisonnalite_12 = np.sum(ssa_components_sla[0, seasonal_modes_12m], axis=0) if seasonal_modes_12m else np.zeros_like(ssa_components_sla[0, 0])
#     saisonnalite_other = np.sum(ssa_components_sla[0, other_seasonal_modes], axis=0) if other_seasonal_modes else np.zeros_like(ssa_components_sla[0, 0])
#     bruit = np.sum(ssa_components_sla[0, noise_modes], axis=0) if noise_modes else np.zeros_like(ssa_components_sla[0, 0])
    
#     # # Tracé des résultats
#     # plt.figure(figsize=(12, 8))
#     # plt.plot(trend_removed.index, trend_removed, label="Signal dé-trendé", alpha=0.5)
#     # plt.plot(trend_removed.index, tendance_nl, label="Tendance non linéaire", color='red', linewidth=2)
#     # plt.plot(trend_removed.index, saisonnalite, label="Saisonnalité (6-12 mois)", color='blue', linestyle="--")
#     # plt.plot(trend_removed.index, bruit, label="Bruit", color='gray', linestyle=":")
    
#     # plt.xlabel("Année")
#     # plt.ylabel("Amplitude")
#     # plt.legend()
#     # plt.title("Décomposition SSA : Tendance - Saisonnalité - Bruit")
#     # plt.grid()
#     # plt.show()
    
#     # Affichage des modes classés
#     # print(f"Modes tendance non linéaire (>10 ans) : {low_freq_modes}")
#     # print(f"Modes saisonniers (6-12 mois) : {seasonal_modes}")
#     # print(f"Modes bruit (haute fréquence) : {noise_modes}")
    
#     adjusted_sla = trend_removed-basse_freq_2_5-basse_freq_5_10
#     sla = adjusted_sla
#     sla = sla.rolling(window=6, min_periods=1).mean()
    
#     # # Tracer les résultats
#     # plt.figure(figsize=(20, 8))
#     # plt.plot(sla_15_26.index , sla_15_26['sla']* 10**3, label="SLA X-TRACK", color='blue')

#     # plt.title("Série temporelle ", fontsize=14)
#     # plt.xlabel('Date', fontsize=12)
#     # plt.ylabel('Hauteur (mm)', fontsize=12)
#     # plt.legend()
#     # plt.grid(True)
#     # # plt.ylim(-100, 100)
#     # # plt.yticks(range(-100, 101, 50))
#     # plt.xticks(rotation=45)
#     # plt.tight_layout()
#     # plt.show()
#     return sla 
 
# param_dict = {i: 45 for i in sla_15_nearby}  
# param_dict[0] =15
# param_dict[1]=40
# param_dict[2]=65
# param_dict[4]=60
# param_dict[5]=55
# param_dict[6]=35
# param_dict[7]=50
# param_dict[8]=48
# param_dict[9]=35 
# param_dict[10]=35
# param_dict[11]=50
# param_dict[15]=65
# param_dict[17]=19 
# param_dict[18]=25 
# param_dict[19]=11
# param_dict[20]=20
# param_dict[21]=20
# param_dict[22]=20
# param_dict[23]=24
# param_dict[24]=37
# param_dict[25]=30
# param_dict[26]=51 
# param_dict[27]=17 
# param_dict[28]=35
# param_dict[29]=34
# param_dict[30]=29
# param_dict[31]=33
# param_dict[32]=33
# param_dict[33]=80
# param_dict[34]=36
# param_dict[35]=37
# param_dict[36]=11 
# param_dict[37]=17
# param_dict[38]=60
# param_dict[39]=55
# param_dict[40]=65
# param_dict[43]=55
# param_dict[44]=44
# param_dict[45]=49
# param_dict[46]=60
# param_dict[47]=53


# ssa_15 = {}
# for i in sla_15_nearby:
#     s = ssa(sla_15_nearby[i], param_dict[i])
#     ssa_15[i] = s 
    
#     # Tracer les résultats
#     plt.figure(figsize=(20, 8))
#     #Tracer toutes les séries sur le même graphique
#     plt.plot(ssa_15[i].index, ssa_15[i] * 10**3, label=f"X-TRACK {idx_15_nearby[i]}")
#     plt.plot(sla_15_26.index , sla_15_26['sla']* 10**3, label="SLA X-TRACK", color='red')
    
#     # Configuration du graphique
#     plt.title("Séries temporelles SSA", fontsize=14)
#     plt.xlabel("Date", fontsize=12)
#     plt.ylabel("Hauteur (mm)", fontsize=12)
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     # Affichage unique du graphique
#     plt.show()

# ssa_df = pd.DataFrame({f"X-TRACK {idx_15_nearby[i]}": ssa_15[i] for i in sla_15_nearby})

# # Supprimer les valeurs NaN si nécessaire
# ssa_df = ssa_df.dropna()

# # Calculer la matrice de corrélation de Pearson
# correlation_matrix = ssa_df.corr(method='pearson')

# # Tracer la heatmap de corrélation
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
# plt.title("Matrice de corrélation de Pearson entre les courbes SSA")
# plt.show()

# # Extraire les données nécessaires
# #ssa_df_annual = ssa_df.resample('M').mean()
# ssa_df.columns = [f"Point {idx}" for idx in idx_15_nearby]

# # Extraire les données mises à jour
# time_values = ssa_df.index.year  # Années en X
# sla_values = ssa_df.values *10**3 # Moyenne annuelle SSA

# # Création de la figure
# plt.figure(figsize=(12, 6))

# # Définir la norme pour centrer le 0 dans le blanc
# divnorm = colors.TwoSlopeNorm(vmin=-400, vcenter=0, vmax=400)

# # Affichage de la carte 2D avec l'échelle de couleur fixée
# plt.pcolormesh(time_values, np.arange(len(ssa_df.columns)), sla_values.T, shading='auto', cmap='coolwarm', norm=divnorm)

# # Ajouter une barre de couleur avec les mêmes limites
# cbar = plt.colorbar(label="Niveau de la mer (mm)")
# cbar.set_ticks(np.arange(-400, 401, 100))  # Ticks tous les 100 mm (ajuste si besoin)

# # Titres et axes
# plt.xlabel("Année")
# plt.ylabel("Indices des points sélectionnés")

# plt.title("Évolution annuelle du niveau de la mer")

# # Modifier les ticks Y pour n'afficher qu'un sous-ensemble des points
# step = 2  # Afficher 1 point sur 2 (ajuste à 3, 4... selon besoin)
# selected_ticks = np.arange(0, len(ssa_df.columns), step)  
# plt.yticks(selected_ticks, ssa_df.columns[selected_ticks])

# # Rotation des dates pour la lisibilité
# plt.xticks(rotation=45)

# # Afficher le graphique
# plt.show()

#%% EOF PENSACOLA, CEDAR KEY II, ST. PETERSBURG, FORT MYERS, NAPLES/ TG ET SEANOE 
# save_path = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\Décomposition"
# #--------TG--------
# # Sélection des stations d'intérêt
# stations = ["PENSACOLA", "CEDAR KEY II", "ST. PETERSBURG","FORT MYERS","NAPLES"]
# # Vérification des clés disponibles
# data_selected = {station: dataframes[station] for station in stations if station in dataframes}

# # S'assurer que les stations existent bien dans dataframes
# if len(data_selected) < len(stations):
#     missing_stations = set(stations) - set(data_selected.keys())
#     print(f"Attention : les stations suivantes ne sont pas présentes dans dataframes : {missing_stations}")

# # Renommer la colonne 'niveau d'eau' pour éviter les conflits et définir 'temps' comme index
# for station, df in data_selected.items():
#     df = df[['temps', "hauteur d'eau en mm"]].copy()  # Sélection des colonnes
#     df = df.rename(columns={"hauteur d'eau en mm": station})  # Renommage de la colonne
#     df = df.set_index('temps')  # Mise en index sur le temps
#     data_selected[station] = df  # Mise à jour dans le dictionnaire

# # Concaténer les DataFrames sur l'index 'temps'
# df_combined = pd.concat(data_selected.values(), axis=1)

# mask= (df_combined.index.year >= 2002) & (df_combined.index.year <= 2021)
# df_combined = df_combined .loc[mask]
# df_combined = df_combined.resample('Y').mean() 

#   #df_combined.to_csv('Pensacola_CedarKey_KeyWest.csv', index=True)

# X = df_combined.to_numpy()
# X= sig.detrend(X, axis=0)
# weights = np.sqrt(np.abs(X) / np.sum(np.abs(X)))

# solver = Eof(X,weights=weights)
# neofs = 3
# eofs = solver.eofsAsCovariance(neofs=neofs)

# eofvar = solver.varianceFraction(neigs=neofs) * 100
# print(f"Variance expliquée par les EOFs : {eofvar}%")

# pcs = solver.pcs(pcscaling=1, npcs=neofs)
# # Reconstruction avec tous les EOFs
# reconstruction = np.dot(eofs.T, pcs.T)  # Résultat sous forme (3, N) où 3 = nombre de stations

# # Reconstruction avec uniquement EOF-1
# reconstruction_eof1 = np.dot(eofs[0, :].reshape(-1, 1), pcs[:, 0].reshape(1, -1))  # Résultat (3, N)

# # Transposer pour obtenir la même forme que df_combined
# df_reconstructed_eof1 = pd.DataFrame(reconstruction_eof1.T, index=df_combined.index, columns=df_combined.columns)
# df_reconstructed_total = pd.DataFrame(reconstruction.T, index=df_combined.index, columns=df_combined.columns)

# for station in df_combined.columns:
#     plt.figure(figsize=(20, 10))  # Crée une nouvelle figure pour chaque station
#     plt.plot(df_reconstructed_eof1.index, df_reconstructed_eof1[station], linestyle="--", label="EOF-1")
#     plt.plot(df_reconstructed_total.index, df_reconstructed_total[station], label="Reconstruction complète")
    
#     plt.legend()
#     plt.title(f"Contribution EOF-1 pour {station}")
#     plt.xlabel("Temps")
#     plt.ylabel("Amplitude")
#     plt.grid()  # Grille sur toute la figure
#     plt.savefig(os.path.join(save_path, f"decomposition avec EOF-TG-{station}.png"), dpi=400)
#     plt.show()  # Afficher chaque figure séparément
    
# #-------Seanoe-------
def open_senoe(i, dataset):    
    s = pd.DataFrame(dataset[i]['sla_mean_10pts'].data, columns=[i])
    t=dataset[i]['time'].data
    s['temps'] = [tref + datetime.timedelta(days=int(j)) for j in t]
    s = s.dropna()
    
    s = s.set_index('temps')
    #s=s.resample('Y').mean() 

    return s 


# txt_file_path = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\liste fichier seanoe.txt"

# with open(txt_file_path, 'r') as f:
#     filenames = f.read().splitlines()

# files = [f"C:\\Users\\camil\\OneDrive\\Bureau\\Univ_de_la_Rochelle\\M2\\semestre 10\\SEANEO\\{filename}" for filename in filenames]

# datasets = [xr.open_dataset(file, decode_times=False, engine='netcdf4') for file in files]

# df_s = []    
# for i, dataset in enumerate(datasets):
#     s = open_senoe(i, datasets)
#     df_s.append(s)
    
# df_combined_s =pd.concat(df_s, axis=1)
# df_combined_s= df_combined_s.interpolate(method='linear', axis=1)
# df_combined_s= df_combined_s.interpolate(method='linear')
# df_mean_curve = df_combined_s.mean(axis=1)

# df_mean_1_to_10 = df_combined_s.iloc[:, 0:10].mean(axis=1)
# df_mean_11_to_22 = df_combined_s.iloc[:, 10:22].mean(axis=1)
# df_mean_23_to_32 = df_combined_s.iloc[:, 22:32].mean(axis=1)


# fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
# # Tracer les données de toutes les stations sur chaque subplot
# for station in df_combined_s.columns:
#     for ax in axes:
#         ax.plot(df_combined_s.index, df_combined_s[station], color='blue', alpha=0.1)  # Un peu de transparence pour éviter le chevauchement

# # Premier subplot : Moyenne globale
# axes[0].plot(df_mean_curve.index, df_mean_curve, color='darkorange')
# axes[0].set_title("Moyenne globale")
# axes[0].set_ylabel("Valeurs")
# axes[0].grid(True)

# # Deuxième subplot : Moyenne des colonnes 1 à 10
# axes[1].plot(df_mean_curve.index, df_mean_curve, color='darkorange')
# axes[1].plot(df_combined_s.index, df_mean_1_to_10, color='green')
# axes[1].set_title("Moyenne des colonnes 1 à 10")
# axes[1].set_ylabel("Valeurs")
# axes[1].grid(True)

# # Troisième subplot : Moyenne des colonnes 11 à 22
# axes[2].plot(df_mean_curve.index, df_mean_curve, color='darkorange')
# axes[2].plot(df_combined_s.index, df_mean_11_to_22, color='purple')
# axes[2].set_title("Moyenne des colonnes 11 à 22")
# axes[2].set_ylabel("Valeurs")
# axes[2].grid(True)

# # Quatrième subplot : Moyenne des colonnes 23 à 32
# axes[3].plot(df_mean_curve.index, df_mean_curve, color='darkorange')
# axes[3].plot(df_combined_s.index, df_mean_23_to_32, color='black')
# axes[3].set_title("Moyenne des colonnes 23 à 32")
# axes[3].set_xlabel("Temps")
# axes[3].set_ylabel("Valeurs")
# axes[3].grid(True)

# # Ajustement des marges pour éviter le chevauchement
# plt.tight_layout()
# plt.show()

# #--eof seanoe--
# S = df_combined_s.to_numpy()
# S= sig.detrend(S, axis=0)
# weights_s = np.sqrt(np.abs(S) / np.sum(np.abs(S)))

# solver_s = Eof(S,weights=weights_s)
# neofs_s = 3
# eofs_s = solver_s.eofsAsCovariance(neofs=neofs_s)

# eofvar_s = solver_s.varianceFraction(neigs=neofs_s) * 100
# print(f"Variance expliquée par les EOFs : {eofvar_s}%")

# pcs_s = solver_s.pcs(pcscaling=1, npcs=neofs_s)
# # Reconstruction avec tous les EOFs
# reconstruction_s = np.dot(eofs_s.T, pcs_s.T)  # Résultat sous forme (3, N) où 3 = nombre de stations

# # Reconstruction avec uniquement EOF-1
# reconstruction_eof1_s = np.dot(eofs_s[0, :].reshape(-1, 1), pcs_s[:, 0].reshape(1, -1))  # Résultat (3, N)

# # Transposer pour obtenir la même forme que df_combined
# df_reconstructed_eof1_s = pd.DataFrame(reconstruction_eof1_s.T, index=df_combined_s.index, columns=df_combined_s.columns)
# df_reconstructed_total_s = pd.DataFrame(reconstruction_s.T, index=df_combined_s.index, columns=df_combined_s.columns)

# for station in df_combined_s.columns:
#     plt.figure(figsize=(20, 10))  # Crée une nouvelle figure pour chaque station
#     plt.plot(df_reconstructed_eof1_s.index, df_reconstructed_eof1_s[station], linestyle="--", label="EOF-1")
#     plt.plot(df_reconstructed_total_s.index, df_reconstructed_total_s[station], label="Reconstruction complète")
    
#     plt.legend()
#     plt.title(f"Contribution EOF-1 et reconstruction complète - {station}")
#     plt.xlabel("Temps")
#     plt.ylabel("Amplitude")
#     plt.grid()  # Grille sur toute la figure
#     #plt.savefig(os.path.join(save_path, f"decomposition avec EOF-Seanoe-point-{station}.png"), dpi=400)
#     plt.show()  # Afficher chaque figure séparément
#%% tableau 2

# Chargement du fichier VLM
VLM_ulr7a = pd.read_csv("VLM ULR7A.txt", sep="  ", header=None)
VLM_ulr7a.set_index(0, inplace=True)
VLM_ulr7a.columns = ["VLM", "Erreur"]

# Chargement des fichiers SEANOE
path = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\SEANEO\*.nc"
files = glob.glob(path)
numeros_recherches = [ "R4_JA_052_05", "R4_JA_193_09","R4_JA_193_01",
                      "R4_JA_026_01","R4_JA_102_02","R4_JA_102_10","R4_JA_102_11"]
matching_files = [f for f in files if any(num in f for num in numeros_recherches)]

datasets = [xr.open_dataset(f, decode_times=False) for f in matching_files]
df_s = [open_senoe(i, datasets) for i, dataset in enumerate(datasets)]

# Concaténation des fichiers SEANOE
seanoe = pd.concat(df_s, axis=1)
nouveaux_noms = ["GALVESTON II", "GRAND ISLE","DAUPHIN ISLAND",
                 "PENSACOLA","ST. PETERSBURG","FORT MYERS","NAPLES"]
seanoe.columns = nouveaux_noms
seanoe.index = pd.to_datetime(seanoe.index)  # Conversion unique

# Initialisation des résultats
resultats = []
stations_to_drop = ["APALACHICOLA", "CEDAR KEY II", "FREEPORT", "GALVESTON I", 
                    "KEY WEST", "PORT ISABEL", "ROCKPORT", "VERACRUZ"]

slopes_tg = []
slopes_tg_long = []
slopes_s= []

dataframes_2 = {key: df for key, df in dataframes.items() if key not in stations_to_drop}

window_sizes_TG = {
    "DAUPHIN ISLAND": 24,
    "FORT MYERS": 24,
    "GALVESTON II": 24,
    "GRAND ISLE": 24,
    "NAPLES": 24,
    "PENSACOLA": 24,
    "ST. PETERSBURG": 24,
}

window_sizes_s = {
    "DAUPHIN ISLAND": 24,
    "FORT MYERS": 30,
    "GALVESTON II": 30,
    "GRAND ISLE": 30,
    "NAPLES": 30,
    "PENSACOLA": 30,
    "ST. PETERSBURG": 24,
}

fs = 1  # Fréquence d'échantillonnage (1 point par mois)
# Calcul des tendances
for key, df in dataframes_2.items():
    print(f"Station: {key}")
    
    # Filtrer les données entre 2002 et 2021
    df_ent= df.copy().dropna()
    mask = (df["temps"].dt.year >= 2002) & (df["temps"].dt.year <= 2021) 
    df = df.loc[mask]
    
    #----TG sur toute la période-----
    if "temps" in df_ent.columns and "hauteur d'eau en mm" in df_ent.columns:
        window_size_tg = window_sizes_TG.get(key, 24)
        ssa_ent = SingularSpectrumAnalysis(window_size=window_size_tg, groups='auto')
        ssa_components_ent = ssa_ent.fit_transform(df_ent["hauteur d'eau en mm"].values.reshape(1, -1))
    
        num_modes_ent = ssa_components_ent.shape[1]  # Nombre de modes
        
        # Analyse spectrale pour trier les modes
        low_freq_modes_ent = []
        for mode in range(num_modes_ent):
            signal_ent = ssa_components_ent[0, mode]
            N = len(signal_ent)
    
            # FFT pour récupérer les fréquences
            freq = np.fft.fftfreq(N, d=1/fs)
            fft_values = np.fft.fft(signal_ent) 
            power_spectrum = np.abs(fft_values)**2
    
            # Périodes associées aux fréquences positives
            positive_freqs = freq[freq > 0]
            positive_spectrum = power_spectrum[freq > 0]
            periods = 1 / positive_freqs  # Conversion en périodes (mois)
    
            # Identifier la période dominante du mode
            if len(periods) > 0:
                peak_period = periods[np.argmax(positive_spectrum)]  # Période dominante
            else:
                peak_period = 0  
    
            # Si la période dominante est >= 24 mois, on garde le mode
            if peak_period >= 24:
                low_freq_modes_ent.append(mode)
        
        # Reconstruction du signal filtré avec seulement les basses fréquences
        filtered_signal_ent = np.sum([ssa_components_ent[0, mode, :] for mode in low_freq_modes_ent], axis=0)
        if filtered_signal_ent.any() == 0 :
            continue
        # # Affichage
        # plt.figure(figsize=(10, 5))
        # plt.plot(df_clean["temps"], df_clean["hauteur d'eau en mm"], alpha=0.5, label="Original")
        # plt.plot(df_clean["temps"], filtered_signal, label="Basses fréquences (SSA)", color="red")
        # plt.legend()
        # plt.title(f"SSA Filtré - {key}")
        # plt.xlabel("Temps")
        # plt.ylabel("Hauteur d'eau (mm)")
        # plt.show()
        # fig, axes = plt.subplots(num_modes, 1, figsize=(10, 2 * num_modes), sharex=True)
        # if num_modes == 1:
        #     axes = [axes]
    
        # for i in range(num_modes):
        #     axes[i].plot( ssa_components[0, i, :], label=f'Mode {i+1}')
        #     axes[i].legend()
    
        # plt.xlabel('Temps')
        # plt.show()
        
        if not df_ent.empty:
            temps_numerique = (df_ent["temps"] - df_ent["temps"].min()).dt.total_seconds() / (365.25 * 24 * 3600)
            slope_ent, intercept, r_value, p_value, std_err_ent = linregress(temps_numerique, filtered_signal_ent )
        else:
            slope, std_err = np.nan, np.nan
        slopes_tg_long.append(slope_ent) 
    #----TG entre 2002 et 2021-----
    if "temps" in df.columns and "hauteur d'eau en mm" in df.columns:
        df_clean = df.dropna(subset=["hauteur d'eau en mm"])
       
        window_size_tg = window_sizes_TG.get(key, 24)
        ssa = SingularSpectrumAnalysis(window_size=window_size_tg, groups='auto')
        ssa_components = ssa.fit_transform(df_clean["hauteur d'eau en mm"].values.reshape(1, -1))

        num_modes = ssa_components.shape[1]  # Nombre de modes
        
        # Analyse spectrale pour trier les modes
        low_freq_modes = []
        for mode in range(num_modes):
            signal = ssa_components[0, mode]
            N = len(signal)

            # FFT pour récupérer les fréquences
            freq = np.fft.fftfreq(N, d=1/fs)
            fft_values = np.fft.fft(signal) 
            power_spectrum = np.abs(fft_values)**2

            # Périodes associées aux fréquences positives
            positive_freqs = freq[freq > 0]
            positive_spectrum = power_spectrum[freq > 0]
            periods = 1 / positive_freqs  # Conversion en périodes (mois)

            # Identifier la période dominante du mode
            if len(periods) > 0:
                peak_period = periods[np.argmax(positive_spectrum)]  # Période dominante
            else:
                peak_period = 0  

            # Si la période dominante est >= 24 mois, on garde le mode
            if peak_period >= 24:
                low_freq_modes.append(mode)
        
        # Reconstruction du signal filtré avec seulement les basses fréquences
        filtered_signal = np.sum([ssa_components[0, mode, :] for mode in low_freq_modes], axis=0)
        if filtered_signal.any() == 0 :
            continue
        # # Affichage
        # plt.figure(figsize=(10, 5))
        # plt.plot(df_clean["temps"], df_clean["hauteur d'eau en mm"], alpha=0.5, label="Original")
        # plt.plot(df_clean["temps"], filtered_signal, label="Basses fréquences (SSA)", color="red")
        # plt.legend()
        # plt.title(f"SSA Filtré - {key}")
        # plt.xlabel("Temps")
        # plt.ylabel("Hauteur d'eau (mm)")
        # plt.show()
        # fig, axes = plt.subplots(num_modes, 1, figsize=(10, 2 * num_modes), sharex=True)
        # if num_modes == 1:
        #     axes = [axes]

        # for i in range(num_modes):
        #     axes[i].plot( ssa_components[0, i, :], label=f'Mode {i+1}')
        #     axes[i].legend()

        # plt.xlabel('Temps')
        # plt.show()
        
        if not df_clean.empty:
            temps_numerique = (df_clean["temps"] - df_clean["temps"].min()).dt.total_seconds() / (365.25 * 24 * 3600)
            slope, intercept, r_value, p_value, std_err = linregress(temps_numerique, filtered_signal )
        else:
            slope, std_err = np.nan, np.nan
        slopes_tg.append(slope) 
        #-----VLM par l'approche direct-----
        if key in VLM_ulr7a.index:
            vlm_value = VLM_ulr7a.loc[key, "VLM"]
            vlm_error = VLM_ulr7a.loc[key, "Erreur"]
        else:
            vlm_value, vlm_error = np.nan, np.nan
        
        TG_VLM = slope - abs(vlm_value) if not np.isnan(slope) and not np.isnan(vlm_value) else np.nan
        TG_VLM_error = np.sqrt(std_err**2 + vlm_error**2) if not np.isnan(std_err) and not np.isnan(vlm_error) else np.nan
        
        TG_VLM_all= slope_ent - abs(vlm_value) if not np.isnan(slope_ent) and not np.isnan(vlm_value) else np.nan
        TG_VLM_error_all = np.sqrt(std_err_ent**2 + vlm_error**2) if not np.isnan(std_err_ent) and not np.isnan(vlm_error) else np.nan
        
        #-----Seanoe-----
        if key in seanoe.columns:
            df_clean_s = seanoe[key].dropna()
            window_size_s = window_sizes_s.get(key, 24)
            ssa_s = SingularSpectrumAnalysis(window_size=window_size_s, groups='auto')
            ssa_components_s = ssa_s.fit_transform(df_clean_s.values.reshape(1, -1))

            num_modes_s = ssa_components_s.shape[1]  # Nombre de modes
            
            # Analyse spectrale pour trier les modes
            low_freq_modes_s = []
            for mode in range(num_modes_s):
                signal = ssa_components_s[0, mode]
                N = len(signal)

                # FFT pour récupérer les fréquences
                freq = np.fft.fftfreq(N, d=1/fs)
                fft_values = np.fft.fft(signal) 
                power_spectrum = np.abs(fft_values)**2

                # Périodes associées aux fréquences positives
                positive_freqs = freq[freq > 0]
                positive_spectrum = power_spectrum[freq > 0]
                periods = 1 / positive_freqs  # Conversion en périodes (mois)

                # Identifier la période dominante du mode
                if len(periods) > 0:
                    peak_period = periods[np.argmax(positive_spectrum)]  # Période dominante
                else:
                    peak_period = 0  

                # Si la période dominante est >= 24 mois, on garde le mode
                if peak_period >= 24:
                    low_freq_modes_s.append(mode)
            
            # Reconstruction du signal filtré avec seulement les basses fréquences
            filtered_signal_s = np.sum([ssa_components_s[0, mode, :] for mode in low_freq_modes_s], axis=0)
            if filtered_signal_s.any() == 0 :
                continue
            # Affichage
            plt.figure(figsize=(10, 5))
            plt.plot(df_clean_s.index, df_clean_s, alpha=0.5, label="Original")
            plt.plot(df_clean_s.index, filtered_signal_s, label="Basses fréquences (SSA)", color="red")
            plt.legend()
            plt.title(f"SSA Filtré - {key}")
            plt.xlabel("Temps")
            plt.ylabel("Hauteur d'eau (mm)")
            plt.show()
            fig, axes = plt.subplots(num_modes_s, 1, figsize=(10, 2 * num_modes_s), sharex=True)
            if num_modes_s == 1:
                axes = [axes]

            for i in range(num_modes_s):
                axes[i].plot( ssa_components_s[0, i, :], label=f'Mode {i+1}')
                axes[i].legend()

            plt.xlabel('Temps')
            plt.show()
            if not df_clean_s.empty:
                temps_numerique_s = (df_clean_s.index - df_clean_s.index.min()).days / 365.25
                slope_s, intercept_s, r_value_s, p_value_s, std_err_s = linregress(temps_numerique_s, filtered_signal_s*10**3)
                
            else:
                slope_s, std_err_s = np.nan, np.nan
        else:
            slope_s, std_err_s = np.nan, np.nan
        slopes_s.append(slope_s) 
        
        # Ajouter les résultats
        resultats.append({
            "Station": key,
            "Tendance linéaire all": f"{round(slope_ent, 2)} ± {round(std_err_ent, 2)}",
            "Tendance linéaire 2002-2021": f"{round(slope, 2)} ± {round(std_err, 2)}",
            "VLM ULR7A": f"{round(vlm_value, 2)} ± {round(vlm_error, 2)}",
            "TG-VLM all": f"{round(TG_VLM_all, 2)} ± {round(TG_VLM_error_all, 2)}",
            "TG-VLM 2002-2021": f"{round(TG_VLM, 2)} ± {round(TG_VLM_error, 2)}",
            "Alti": f"{round(slope_s, 2)} ± {round(std_err_s, 2)}"
        })

# Création du DataFrame final
resultats_df = pd.DataFrame(resultats)
resultats_df = resultats_df[~resultats_df["Station"].isin(stations_to_drop)]
resultats_df.set_index("Station", inplace=True)

# #%%  tendances dans le marégraphes - fenêtres de 20 Ans
# save_path = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\Tendance TG"
# #----tendance OLS----
#     #---20 ans---
# for key, df in dataframes.items():

#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Filtrer les données pour chaque période
#     periods = [(2002, 2011), (2012, 2021)]
#     colors = ['red', 'green']  # Couleurs des tendances
    
#     for i, (start, end) in enumerate(periods):
#         mask_tg = (df['temps'].dt.year >= start) & (df['temps'].dt.year <= end)
#         mask_tg_2 = (df['temps'].dt.year >= 2002) & (df['temps'].dt.year <= 2021)

#         df_filtered = df.loc[mask_tg].copy().dropna()
        
#         time_numeric = ((df_filtered['temps'] - df_filtered['temps'].min()).dt.total_seconds().values)

#         if not df_filtered.empty:
#             # Variables pour OLS
#             X = sm.add_constant(time_numeric )  # Ajouter une constante pour l'intercept
#             y = df_filtered["hauteur d'eau en mm"]  # Remplace 'valeur' par le bon nom de colonne
    
#             # OLS
#             X = sm.add_constant(time_numeric)
#             y = df_filtered["hauteur d'eau en mm"]
#             model = sm.OLS(y, X).fit()
#             trend = model.predict(X)

#             # Extraire pente (en mm/an) et erreur standard
#             slope = model.params[1]* 365.25 * 24 * 3600
#             slope_std = model.bse[1]* 365.25 * 24 * 3600

#             # Tracer la tendance
#             ax.plot(df_filtered['temps'], trend, color=colors[i], label=f"Pente: {slope:.2f} ± {slope_std:.2f} mm/an")

#     # Tracer les valeurs réelles
#     df = df.loc[mask_tg_2].dropna()
#     ax.plot(df['temps'], df["hauteur d'eau en mm"], color='blue', alpha=0.5, label='Données brutes')
#     # Mise en forme
#     ax.set_title(f"Tendances pour {key}")
#     ax.set_xlabel("Année")
#     ax.set_ylabel("Hauteur d'eau (mm)")
#     ax.legend()
#     ax.grid()

#     plt.tight_layout()
#     #plt.savefig(os.path.join(save_path, f"Tendance OLS {key}.png"), dpi=400)
#     plt.show()

# #----Tendance OLS après application du SSA----
# # Paramètres SSA
# window_size =12  # taille de fenêtre (en nombre de points, à ajuster selon ta résolution)
# ssa = SingularSpectrumAnalysis(window_size=window_size)

# for key, df in dataframes.items():

#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     periods = [(2002, 2011), (2012, 2021)]
#     colors = ['red', 'green']
    
#     for i, (start, end) in enumerate(periods):
#         mask = (df['temps'].dt.year >= start) & (df['temps'].dt.year <= end)
#         df_filtered = df.loc[mask].copy().dropna()

#         if df_filtered.empty:
#             continue

#         # SSA pour extraction de la tendance
#         y = df_filtered["hauteur d'eau en mm"].values.reshape(1, -1)
#         component = ssa.fit_transform(y)  # On prend le premier composant (souvent la tendance)
#         trend_component=component[0, 0]
#         # Temps numérique pour OLS
#         time_numeric = (df_filtered['temps'] - df_filtered['temps'].min()).dt.total_seconds().values
#         X = sm.add_constant(time_numeric)
#         model = sm.OLS(trend_component, X).fit()
#         trend = model.predict(X)

#         # Calcul de pente en mm/an
#         slope = model.params[1] * 365.25 * 24 * 3600
#         slope_std = model.bse[1] * 365.25 * 24 * 3600

#         # Tracé
#         #ax.plot(df_filtered['temps'], trend_component, label=f'Tendance SSA {start}-{end}', color=colors[i])
#         ax.plot(df_filtered['temps'], trend, linestyle='--', color=colors[i], label=f"Pente: {slope:.2f} ± {slope_std:.2f} mm/an")

#     # Données brutes
#     mask_full = (df['temps'].dt.year >= 2002) & (df['temps'].dt.year <= 2021)
#     df_full = df.loc[mask_full].dropna()
#     ax.plot(df_full['temps'], df_full["hauteur d'eau en mm"], color='blue', alpha=0.4, label='Données brutes')

#     # Légendes et mise en forme
#     ax.set_title(f"Tendance : {key}")
#     ax.set_xlabel("Année")
#     ax.set_ylabel("Hauteur (mm)")
    
#     ax.legend()
#     ax.grid()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f"Tendance SSA_OLS {key}.png"), dpi=400)
#     plt.show()

# #----Tendance OLS après application de l'EOF----
# # Étape 1 : construire la matrice temps x stations
# data_matrix = pd.concat(
#     [df.set_index('temps')["hauteur d'eau en mm"] for df in dataframes.values()],
#     axis=1
# )
# data_matrix.columns = list(dataframes.keys())
# mask_tg_2 = (data_matrix.index.year >= 2002) & (data_matrix.index.year <= 2021)
# data_matrix = data_matrix.loc[mask_tg_2]
# data_matrix = data_matrix.interpolate(method='linear', axis=0)
# w=np.sqrt(np.abs(data_matrix) / np.sum(np.abs(data_matrix)))

# # Étape 2 : appliquer EOF
# solver = Eof(data_matrix.values,weights=w)
# neofs=3
# eofs = solver.eofsAsCovariance(neofs=neofs)

# eofvar = solver.varianceFraction(neigs=neofs) * 100
# print(f"Variance expliquée par les EOFs : {eofvar}%")

# pc1 = solver.pcs(npcs=neofs, pcscaling=1)
# reconstruction = np.dot(eofs[0, :].reshape(-1, 1), pc1[:, 0].reshape(1, -1))
# trend_df = pd.DataFrame(reconstruction.T, index=data_matrix.index, columns=data_matrix.columns)

# for station in data_matrix.columns:
#     series_brute = data_matrix[station]
#     trend_station = trend_df[station]

#     # Sous-séries temporelles
#     trend_1 = trend_station['2002':'2011']
#     trend_2 = trend_station['2012':'2021']

#     # Fonction utilitaire pour faire l'OLS sur une période
#     def fit_ols(serie):
#         temps = (serie.index - serie.index.min()).total_seconds()
#         X = sm.add_constant(temps)
#         model = sm.OLS(serie.values, X).fit()
#         slope = model.params[1] * 365.25 * 24 * 3600  # mm/an
#         slope_std = model.bse[1] * 365.25 * 24 * 3600
#         return model.predict(X), slope, slope_std, serie.index

#     # Tendance 2002–2011
#     pred_1, pente_1, std_1, index_1 = fit_ols(trend_1)

#     # Tendance 2012–2021
#     pred_2, pente_2, std_2, index_2 = fit_ols(trend_2)

#     # Tracé
#     plt.figure(figsize=(10, 4))
#     plt.plot(series_brute.index, trend_station, label='Série brute', color='lightgray')
#     plt.plot(index_1, pred_1, label=f'Tendance 2002–2011\n{pente_1:.2f} ± {std_1:.2f} mm/an', color='blue', linewidth=2)
#     plt.plot(index_2, pred_2, label=f'Tendance 2012–2021\n{pente_2:.2f} ± {std_2:.2f} mm/an', color='red', linewidth=2)
#     plt.title(f'Station : {station}')
#     plt.xlabel('Temps')
#     plt.ylabel("Hauteur d'eau (mm)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f"Tendance EOF_OLS {station}.png"), dpi=400)
#     plt.show()
#%% figure 1 Oelsmann_2024 / figure 1 avec EOF TG et Seanoe 

#----TG----
window_sizes_TG = {
    "KEY WEST":22,
    "CEDAR KEY II":24,
    "FORT MYERS": 24,
    "NAPLES": 24,
    "PENSACOLA": 24,
    "ST. PETERSBURG":24,
    "APALACHICOLA":24,
    "DAUPHIN ISLAND":24,
    "GRAND ISLE":24,
    "GALVESTON II":24,
    "FREEPORT": 24,
    "ROCKPORT":24
}

stations = ["KEY WEST","NAPLES","FORT MYERS","ST. PETERSBURG","CEDAR KEY II","APALACHICOLA","PENSACOLA","DAUPHIN ISLAND","GRAND ISLE","GALVESTON II"
            ,"FREEPORT","ROCKPORT"]
data_selected = {station: dataframes[station] for station in stations if station in dataframes}

if len(data_selected) < len(stations):
    missing_stations = set(stations) - set(data_selected.keys())
    print(f"Attention : les stations suivantes ne sont pas présentes dans dataframes : {missing_stations}")


for station, df in data_selected.items():

    df = df[['temps', "hauteur d'eau en mm"]].copy() 
    df = df.rename(columns={"hauteur d'eau en mm": station}) 
    df = df.set_index('temps') 
    data_selected[station] = df  
    
df_combined = pd.concat(data_selected.values(), axis=1)
df_combined_all=pd.concat(data_selected.values(), axis=1)
mask= (df_combined.index.year >= 2002) & (df_combined.index.year <= 2021)
df_combined = df_combined .loc[mask]
df_combined= df_combined.interpolate(method='linear')
df_combined= df_combined.resample('M').mean()

TG_d = pd.DataFrame(index=df_combined.index)

for column in df_combined.columns:
    fig, ax = plt.subplots(figsize=(12, 5))

    data = df_combined[column]
    detrended = sig.detrend(data.to_numpy())

    window_size_tg = window_sizes_TG.get(column)
    ssa = SingularSpectrumAnalysis(window_size=window_size_tg, groups='auto')
    ssa_components = ssa.fit_transform(detrended.reshape(1, -1))

    num_modes = ssa_components.shape[1]
    fs = 1
    low_freq_modes = []
    high_freq_modes = []

    for mode in range(num_modes):
        signal = ssa_components[0, mode]
        N = len(signal)

        freq = np.fft.fftfreq(N, d=1/fs)
        fft_values = np.fft.fft(signal)
        power_spectrum = np.abs(fft_values)**2

        positive_freqs = freq[freq > 0]
        positive_spectrum = power_spectrum[freq > 0]
        periods = 1 / positive_freqs

        peak_period = periods[np.argmax(positive_spectrum)]

        if peak_period >= 24:
            low_freq_modes.append(mode)
        else:
            high_freq_modes.append(mode)
    print(high_freq_modes)
    hig_s = np.sum(ssa_components[0, high_freq_modes], axis=0)
    detrended = detrended - hig_s

    TG_d[column] = detrended

    ax.plot(df_combined.index, detrended * 1e-3)

    ax.set_title(f'Station {column}')

    ax.set_ylabel('Valeur (m)')
    plt.tight_layout()
    plt.show()

 #---EOF TG---
X = TG_d.to_numpy()
weights = np.sqrt(np.abs(X) / np.sum(np.abs(X)))

solver = Eof(X,weights=weights)
neofs = 4
eofs = solver.eofsAsCovariance(neofs=neofs)
weights_eof1_tg = eofs[0, :]
eofvar = solver.varianceFraction(neigs=neofs) * 100
print(f"Variance expliquée par les EOFs : {eofvar}%")

pcs = solver.pcs(pcscaling=1, npcs=neofs)

# Reconstruction avec uniquement EOF-1
reconstruction_eof1 = np.dot(eofs[0, :].reshape(-1, 1), pcs[:, 0].reshape(1, -1))  # Résultat (3, N)
reconstruction_eof2 = np.dot(eofs[1, :].reshape(-1, 1), pcs[:, 1].reshape(1, -1))  # Résultat (3, N)
reconstruction = reconstruction_eof1 + reconstruction_eof2  # Résultat sous forme (3, N) où 3 = nombre de stations

# Transposer pour obtenir la même forme que df_combined
df_reconstructed_eof1 = pd.DataFrame(reconstruction_eof1.T, index=df_combined.index, columns=df_combined.columns)
df_reconstructed_eof2 = pd.DataFrame(reconstruction_eof2.T, index=df_combined.index, columns=df_combined.columns)
df_reconstructed_total = pd.DataFrame(reconstruction .T, index=df_combined.index, columns=df_combined.columns)

n_stations = df_reconstructed_total.shape[1]
fig, axes = plt.subplots(n_stations, 1, figsize=(10, 5*n_stations))
if n_stations == 1:
    axes = [axes]

for i, column in enumerate(df_reconstructed_total.columns):
    ax = axes[i] 
    
    data =df_reconstructed_total[column]

    ax.fill_between(df_reconstructed_total.index, data*10**-3, where=(data >= 0), color='red', alpha=0.5)
    ax.fill_between(df_reconstructed_total.index, data*10**-3, where=(data < 0), color='blue', alpha=0.5)
    
    # Ajouter un titre et personnaliser les axes
    ax.set_title(f'EOF après EOF station {column}')
    #ax.set_xlabel('Temps')
    #ax.set_ylabel('Niveau d\'eau (dessaisonalisation + détrendé)')


plt.tight_layout()
plt.show()
#----Seanoe----
save_path = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\Figure 1 Oelsmann"
window_sizes_S= {
    "23": 30,
    "26": 30,
    "27": 30,
    "28": 30,
    "29": 30,
    "24": 30,
    "22": 30,
    "16": 30,
    "13": 30,
    "12": 30,
    "10": 30,
    '11':30,
    '30': 30
}
correspondance = {
    'KEY WEST':'29',
    'NAPLES': '28',
    'FORT MYERS': '27',
    'ST. PETERSBURG': '26',
    'CEDAR KEY II': '25',
    'APALACHICOLA': '23',
    'PENSACOLA': '22',
    'DAUPHIN ISLAND': '21',
    'GRAND ISLE': '15',
    'GALVESTON II': '12',
    'FREEPORT': '11',
    'ROCKPORT': '10'
}
df_sla_corrige= df_sla_corrige.interpolate(method='linear', axis=1)
df_sla_corrige= df_sla_corrige.interpolate(method='linear') 

colonnes_interet = [29,28,27,26,25,23,22,21,15,12,11,10]

serie = df_sla_corrige[colonnes_interet]
S_d= pd.DataFrame(index=serie.index)

for tg_col, seanoe_col in correspondance.items():
    plt.figure(figsize=(20, 10))
    legend_labels = []

    # Seanoe
    if int(seanoe_col) in serie.columns:
        s = serie[int(seanoe_col)]
        detrended = sig.detrend(s.to_numpy())
        window_size_s = int(window_sizes_S.get(str(seanoe_col), 30))  
        ssa = SingularSpectrumAnalysis(window_size=window_size_s, groups='auto')
        ssa_components = ssa.fit_transform(detrended.reshape(1, -1))

        fs = 1
        low_freq_modes = []
        high_freq_modes = []

        for mode in range(ssa_components.shape[1]):
            signal = ssa_components[0, mode]
            N = len(signal)
            freq = np.fft.fftfreq(N, d=1/fs)
            power_spectrum = np.abs(np.fft.fft(signal))**2
            positive_freqs = freq[freq > 0]
            positive_spectrum = power_spectrum[freq > 0]
            periods = 1 / positive_freqs
            peak_period = periods[np.argmax(positive_spectrum)]

            if peak_period >= 24:
                low_freq_modes.append(mode)
            else:
                high_freq_modes.append(mode)
        print(high_freq_modes)
        hig_s = np.sum(ssa_components[0, high_freq_modes], axis=0)
        detrended = detrended - hig_s
        S_d[int(seanoe_col)] = detrended
        plt.fill_between(df_sla_corrige.index, detrended, where=(detrended >= 0), color='red', alpha=0.5, label="Seanoe +")
        plt.fill_between(df_sla_corrige.index, detrended, where=(detrended < 0), color='blue', alpha=0.5, label="Seanoe -")
        legend_labels.extend(["Seanoe +", "Seanoe -"])
    else:
        print(f"[⚠️] Colonne Seanoe {seanoe_col} introuvable.")

    # TG
    if tg_col in df_reconstructed_total.columns:
        data = df_combined[tg_col]
        detrended = sig.detrend(data.to_numpy())
        window_size_tg = window_sizes_TG.get(tg_col)
        ssa = SingularSpectrumAnalysis(window_size=window_size_tg, groups='auto')
        ssa_components = ssa.fit_transform(detrended.reshape(1, -1))

        low_freq_modes = []
        high_freq_modes = []

        for mode in range(ssa_components.shape[1]):
            signal = ssa_components[0, mode]
            N = len(signal)
            freq = np.fft.fftfreq(N, d=1/fs)
            power_spectrum = np.abs(np.fft.fft(signal))**2
            positive_freqs = freq[freq > 0]
            positive_spectrum = power_spectrum[freq > 0]
            periods = 1 / positive_freqs
            peak_period = periods[np.argmax(positive_spectrum)]

            if peak_period >= 24:
                low_freq_modes.append(mode)
            else:
                high_freq_modes.append(mode)

        hig_s = np.sum(ssa_components[0, high_freq_modes], axis=0)
        detrended = detrended - hig_s
        TG_d[tg_col] = detrended

        plt.fill_between(df_combined.index, detrended * 1e-3, where=(detrended >= 0), color='green', alpha=0.5, label="TG +")
        plt.fill_between(df_combined.index, detrended * 1e-3, where=(detrended < 0), color='orange', alpha=0.5, label="TG -")
        legend_labels.extend(["TG +", "TG -"])
    else:
        print(f"[⚠️] Colonne TG {tg_col} introuvable.")

    plt.legend()
    plt.title(f'Station {tg_col}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'Série temporelle TG et Seanoe_detrend_desaisonnalite_SSA_{tg_col}.png'), dpi=400)
    plt.show()



#---EOF Seanoe---
S =S_d.to_numpy()
weights_s = np.sqrt(np.abs(S) / np.sum(np.abs(S)))

solver_s = Eof(S,weights=weights_s)
neofs_s = 4
eofs_s = solver_s.eofsAsCovariance(neofs=neofs_s)
weights_eof1_seanoe = eofs_s[0, :]
eofvar_s = solver_s.varianceFraction(neigs=neofs_s) * 100
print(f"Variance expliquée par les EOFs : {eofvar_s}%")

pcs_s = solver_s.pcs(pcscaling=1, npcs=neofs_s)
# Reconstruction avec tous les EOFs

# Reconstruction avec uniquement EOF-1
reconstruction_eof1_s = np.dot(eofs_s[0, :].reshape(-1, 1), pcs_s[:, 0].reshape(1, -1))  # Résultat (3, N)
reconstruction_eof2_s = np.dot(eofs_s[1, :].reshape(-1, 1), pcs_s[:, 1].reshape(1, -1))  # Résultat (3, N)
reconstruction_eof3_s = np.dot(eofs_s[2, :].reshape(-1, 1), pcs_s[:, 2].reshape(1, -1))  # Résultat (3, N)
reconstruction_s = reconstruction_eof1_s+reconstruction_eof2_s

# Transposer pour obtenir la même forme que df_combined
df_reconstructed_eof1_s = pd.DataFrame(reconstruction_eof1_s.T, index=S_d.index, columns=S_d.columns)
df_reconstructed_total_s = pd.DataFrame(reconstruction_s.T, index=S_d.index, columns=S_d.columns)

for tg_col, seanoe_col in correspondance.items():
    plt.figure(figsize=(20, 10))

    # Seanoe
    if int(seanoe_col) in df_reconstructed_total_s.columns:
        data = df_reconstructed_total_s[int(seanoe_col)]
        plt.fill_between(df_reconstructed_total_s.index, data, where=(data >= 0), color='red', alpha=0.5, label="Seanoe +")
        plt.fill_between(df_reconstructed_total_s.index, data, where=(data < 0), color='blue', alpha=0.5, label="Seanoe -")
    else:
        print(f"Colonne Seanoe {seanoe_col} introuvable.")
    # TG
    if tg_col in df_reconstructed_total.columns:
        d = df_reconstructed_total[tg_col]
        plt.fill_between(df_reconstructed_total.index, d * 1e-3, where=(d >= 0), color='green', alpha=0.5, label="TG +")
        plt.fill_between(df_reconstructed_total.index, d * 1e-3, where=(d < 0), color='orange', alpha=0.5, label="TG -")

    plt.title(f'Après EOF - Station {tg_col}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'Série temporelle_TG_Seanoe_EOF_{tg_col}.png'), dpi=400)
    plt.show()

trend_data = []

for tg_col, seanoe_col in correspondance.items():
    trend_info = {"station": tg_col}
    plt.figure(figsize=(20, 10))

    # Seanoe
    if int(seanoe_col) in df_reconstructed_total_s.columns:
        data = df_reconstructed_total_s[int(seanoe_col)]
        
        # Appliquer LOESS (Local Polynomial Regression) pour la tendance
        loess_result = lowess(data, df_reconstructed_total_s.index, frac=0.1)
        trend_seanoe = loess_result[:, 1]
        residuals_seanoe = data - trend_seanoe
        
        # Calcul de la pente en mm/an
        time_diff = df_reconstructed_total_s.index[-1] - df_reconstructed_total_s.index[0]  # Durée totale
        time_diff_days = time_diff.days  # Différence en jours
        trend_seanoe_slope = (trend_seanoe[-1] - trend_seanoe[0]) / time_diff_days * 1e3 * 365  # mm/an
        std_residuals_seanoe = np.std(residuals_seanoe)
        error_seanoe = (std_residuals_seanoe / np.sqrt(len(data)))*1e3
        
        trend_info["trend_seanoe_slope"] = trend_seanoe_slope
        trend_info["error_seanoe"] = error_seanoe
        
        plt.fill_between(df_reconstructed_total_s.index, data*1e3, where=(data >= 0), color='red', alpha=0.5, label="Seanoe +")
        plt.fill_between(df_reconstructed_total_s.index, data*1e3, where=(data < 0), color='blue', alpha=0.5, label="Seanoe -")
        plt.plot(df_reconstructed_total_s.index, trend_seanoe*1e3, color='pink', label=f"Tendance Seanoe (LOESS) : {trend_seanoe_slope:.2f}  ± {error_seanoe:.2f} mm/an")

    else:
        print(f"Colonne Seanoe {seanoe_col} introuvable.")

    # TG
    if tg_col in df_reconstructed_total.columns:
        d = df_reconstructed_total[tg_col]
        
        # Appliquer LOESS pour la tendance
        loess_result_tg = lowess(d, df_reconstructed_total.index, frac=0.1)
        trend_tg = loess_result_tg[:, 1]
        residuals_tg = d - trend_tg
        
        # Calcul de la pente en mm/an
        time_diff_tg = df_reconstructed_total.index[-1] - df_reconstructed_total.index[0]  # Durée totale
        time_diff_days_tg = time_diff_tg.days  # Différence en jours
        trend_tg_slope = (trend_tg[-1] - trend_tg[0]) / time_diff_days_tg  * 365  # mm/an
        std_residuals_tg = np.std(residuals_tg)
        error_tg = std_residuals_tg / np.sqrt(len(d)) 
        
        trend_info["trend_tg_slope"] = trend_tg_slope
        trend_info["error_tg"] = error_tg

        plt.fill_between(df_reconstructed_total.index, d , where=(d >= 0), color='green', alpha=0.5, label="TG +")
        plt.fill_between(df_reconstructed_total.index, d , where=(d < 0), color='orange', alpha=0.5, label="TG -")
        plt.plot(df_reconstructed_total.index, trend_tg , color='black', label=f"Tendance TG (LOESS) : {trend_tg_slope:.2f} ± {error_tg:.2f} mm/an")
        
    plt.title(f'Après EOF - Station {tg_col}')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join(save_path, f'Série temporelle_TG_Seanoe_EOF_{tg_col}.png'), dpi=400)
    plt.show()
    trend_data.append(trend_info)

trend_df = pd.DataFrame(trend_data)
trend_df.set_index('station', inplace=True)
trend_df["VLM (mm/an)"] = trend_df["trend_seanoe_slope"]- trend_df["trend_tg_slope"] 
trend_df["VLM error"] = np.sqrt(trend_df["error_seanoe"]**2+trend_df["error_tg"]**2  )

# Création de colonnes combinées avec tendance ± erreur
trend_df["Seanoe (mm/an)"] = trend_df.apply(
    lambda row: f"{row['trend_seanoe_slope']:.2f} ± {row['error_seanoe']:.2f}" 
    if not pd.isnull(row['trend_seanoe_slope']) else "NaN", axis=1)

trend_df["TG (mm/an)"] = trend_df.apply(
    lambda row: f"{row['trend_tg_slope']:.2f} ± {row['error_tg']:.2f}" 
    if not pd.isnull(row['trend_tg_slope']) else "NaN", axis=1)

trend_df["VLM (mm/an)"] = trend_df.apply(
    lambda row: f"{row['VLM (mm/an)']:.2f} ± {row['VLM error']:.2f}" 
    if not pd.isnull(row['VLM (mm/an)']) else "NaN", axis=1)


# Garde uniquement les colonnes formatées
trend_df_final = trend_df[["Seanoe (mm/an)", "TG (mm/an)","VLM (mm/an)"]]
fichier_excel = os.path.join(save_path, "tendances_Seanoe_TG_VLM.xlsx")
trend_df_final.to_excel(fichier_excel, index=True)

df_coords_TG = pd.DataFrame(coordonnees).T
df_coords_TG .columns = ['lat', 'lon']
df_coords_TG =df_coords_TG .drop(['VERACRUZ','PORT ISABEL','GALVESTON I'])
df_coords_TG  = df_coords_TG .iloc[::-1]
df_coords_TG ['weights_eof1_tg'] = weights_eof1_tg


# Tracer les poids
plt.figure(figsize=(20, 10))
plt.plot(df_coords_TG.index, weights_eof1_seanoe, label="Poids EOF1 Seanoe", color='blue')
plt.plot(df_coords_TG.index, weights_eof1_tg*1e-3, label="Poids EOF1 TG", color='green')

plt.title("Poids des EOF1 pour Seanoe et TG")
plt.xlabel("Temps")
plt.ylabel("Poids")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% EMD 

def compute_trend(x, y, frac=0.1):
    # Appliquer LOESS (LOWESS)
    x_num = np.arange(len(x))
    loess_result = lowess(y, x_num, frac=frac, return_sorted=False)

    # Résidus
    residuals = y - loess_result

    # Calcul de la pente moyenne sur toute la durée
    time_diff = x[-1] - x[0]
    time_diff_days = time_diff.days
    slope_mm_per_day = (loess_result[-1] - loess_result[0]) / time_diff_days
    slope_mm_per_year = slope_mm_per_day * 365

    # Erreur
    std_residuals = np.std(residuals)
    error = std_residuals / np.sqrt(len(y))

    return loess_result, slope_mm_per_year, error

save_path = r"C:\Users\camil\OneDrive\Bureau\Univ_de_la_Rochelle\M2\semestre 10\EMD"
trend_data_EMD = []    
for tg_col, seanoe_col in correspondance.items():
    fig, ax = plt.subplots(3, 1, figsize=(20, 10))

    # Seanoe
    if int(seanoe_col) in S_d.columns:
        data_s = S_d[int(seanoe_col)].to_numpy()
        data_s = sig.detrend(data_s)
        
        # Appliquer EMD
        emd_s = EMD()
        imfs_s = emd_s(data_s)  # Décomposition en IMF
        
        # Appliquer la FFT sur chaque IMF et isoler les basses fréquences
        low_freq_imfs_s = []
        h_freq_imfs_s = []
        for i, imf in enumerate(imfs_s):
            # Appliquer la FFT
            N = len(imf)
            freqs = np.fft.fftfreq(N)  
            fft_values = np.fft.fft(imf)  
            power_spectrum = np.abs(fft_values)  
        
            cutoff_frequency = 1 / 24
            fft_values[np.abs(freqs) > cutoff_frequency] = 0
            filtered_imf = np.fft.ifft(fft_values).real
            low_freq_imfs_s.append(filtered_imf)
            
            fft_values[np.abs(freqs) < cutoff_frequency] = 0
            filtered_himf = np.fft.ifft(fft_values).real
            h_freq_imfs_s.append(filtered_himf)
        
        # Reconstituer le signal avec les basses fréquences
        reconstructed_signal_s = np.sum(low_freq_imfs_s, axis=0)

    # TG
    if tg_col in TG_d.columns:
        
        data = TG_d[tg_col].to_numpy()
        data = sig.detrend(data)

        emd = EMD()
        imfs = emd(data) 
        
        low_freq_imfs = []
        for i, imf in enumerate(imfs):
            # Appliquer la FFT
            N = len(imf)
            freqs = np.fft.fftfreq(N)  
            fft_values = np.fft.fft(imf)  
            power_spectrum = np.abs(fft_values)  

            cutoff_frequency = 1 / 24
            fft_values[np.abs(freqs) > cutoff_frequency] = 0
            filtered_imf = np.fft.ifft(fft_values).real
            low_freq_imfs.append(filtered_imf)
        
        reconstructed_signal = np.sum(low_freq_imfs, axis=0)
    
    if tg_col in df_combined_all.columns:
        n=df_combined_all[tg_col].dropna()
        data_all = n.to_numpy()
        data_all = sig.detrend(data_all)

        emd = EMD()
        imfs = emd(data_all) 
        
        low_freq_imfs = []
        for i, imf in enumerate(imfs):
            # Appliquer la FFT
            N = len(imf)
            freqs = np.fft.fftfreq(N)  
            fft_values = np.fft.fft(imf)  
            power_spectrum = np.abs(fft_values)  

            cutoff_frequency = 1 / 24
            fft_values[np.abs(freqs) > cutoff_frequency] = 0
            filtered_imf = np.fft.ifft(fft_values).real
            low_freq_imfs.append(filtered_imf)
        
        reconstructed_signal_n = np.sum(low_freq_imfs, axis=0)
    
        # Plot du signal original
        ax[0].plot(df_combined.index, data, label="Signal original")
        ax[0].plot(serie.index, data_s*1e3, label="Signal original_S",color='red')
        ax[0].set_title(f'Signal original - {tg_col}')
        ax[0].legend(loc='upper right')

        # Plot du signal reconstruit avec les basses fréquences
        ax[1].plot(df_combined.index, reconstructed_signal, label="Signal reconstruit_TG (Basses Fréquences)")
        trend_tg_low, slope_tg_low, err_tg_low = compute_trend(df_combined.index, reconstructed_signal)
        ax[1].plot(df_combined.index, trend_tg_low, '--', color='blue', label=f"Trend TG LF : {slope_tg_low:.2f} ± {err_tg_low:.2f} mm/an")
        ax[1].plot(serie.index, reconstructed_signal_s*1e3, label="Signal reconstruit_S (Basses Fréquences)",color='red')
        trend_s_low, slope_s_low, err_s_low = compute_trend(serie.index, reconstructed_signal_s * 1e3)
        ax[1].plot(serie.index, trend_s_low, '--', color='red', label=f"Trend Seanoe LF : {slope_s_low:.2f} ± {err_s_low:.2f} mm/an")
        ax[1].set_title('Signal reconstruit avec Basses Fréquences')
        ax[1].legend(loc='upper right')
        
        ax[2].plot(n.index, reconstructed_signal_n, label="Signal reconstruit_TG (Basses Fréquences)")
        trend_tg_n, slope_tg_n, err_tg_n = compute_trend(n.index, reconstructed_signal_n )
        ax[2].plot(n.index, trend_tg_n, '--', color='blue', label=f"Trend TG All : {slope_tg_n:.2f} ± {err_tg_n:.2f} mm/an")
        ax[2].plot(serie.index, reconstructed_signal_s*1e3, label="Signal reconstruit_S (Basses Fréquences)",color='red')
        trend_s_low2, slope_s_low2, err_s_low2 = compute_trend(serie.index, reconstructed_signal_s * 1e3)
        ax[2].plot(serie.index, trend_s_low2, '--', color='red', label=f"Trend Seanoe LF : {slope_s_low2:.2f} ± {err_s_low2:.2f} mm/an")
        ax[2].set_title('Signal reconstruit avec Basses Fréquences')
        ax[2].legend(loc='upper right')

    plt.legend()
    #plt.title(f'Station {tg_col}')
    #plt.savefig(os.path.join(save_path, f'EMD_{tg_col}.png'), dpi=400)
    plt.tight_layout()
    plt.show()
    trend_data_EMD.append({
        "Station": tg_col,
        "Trend_TG_LF": slope_tg_low,
        "Error_TG_LF": err_tg_low,
        "Trend_Seanoe_LF": slope_s_low,
        "Error_Seanoe_LF": err_s_low,
        "Trend_TG_All": slope_tg_n,
        "Error_TG_All": err_tg_n,
    })


trend_df_EMD = pd.DataFrame(trend_data_EMD)
trend_df_EMD.set_index('Station', inplace=True)
trend_df_EMD["VLM (mm/an)"] = trend_df_EMD["Trend_Seanoe_LF"]- trend_df_EMD["Trend_TG_LF"] 
trend_df_EMD["VLM error"] = np.sqrt(trend_df_EMD["Error_Seanoe_LF"]**2+trend_df_EMD["Error_TG_LF"]**2  )

trend_df_EMD["Seanoe (mm/an)"] = trend_df_EMD.apply(
    lambda row: f"{row['Trend_Seanoe_LF']:.2f} ± {row['Error_Seanoe_LF']:.2f}" 
    if not pd.isnull(row['Trend_Seanoe_LF']) else "NaN", axis=1)

trend_df_EMD["TG (mm/an)"] = trend_df_EMD.apply(
    lambda row: f"{row['Trend_TG_LF']:.2f} ± {row['Error_TG_LF']:.2f}" 
    if not pd.isnull(row['Trend_TG_LF']) else "NaN", axis=1)

trend_df_EMD["VLM (mm/an)"] = trend_df_EMD.apply(
    lambda row: f"{row['VLM (mm/an)']:.2f} ± {row['VLM error']:.2f}" 
    if not pd.isnull(row['VLM (mm/an)']) else "NaN", axis=1)

trend_df_EMD["TG_all (mm/an)"] = trend_df_EMD.apply(
    lambda row: f"{row['Trend_TG_All']:.2f} ± {row['Error_TG_All']:.2f}" 
    if not pd.isnull(row['Trend_TG_All']) else "NaN", axis=1)

# Garde uniquement les colonnes formatées
trend_df_final_EMD = trend_df_EMD[["Seanoe (mm/an)", "TG (mm/an)","VLM (mm/an)","TG_all (mm/an)"]]
#---ensemble EMD---
 #--TG--    
for column in TG_d.columns:
    if column == 'PENSACOLA':
        data = TG_d[column].to_numpy()
        data = data - np.mean(data)
        # Appliquer EMD
        emd = EMD()
        imfs = emd(data)
        n_imfs = len(imfs)
    
        # Calcul de la variance expliquée
        total_variance = np.var(data)
        explained_variances = [np.var(imf) / total_variance for imf in imfs]
    
        # Préparer la figure avec 2 colonnes : IMF et spectre
        fig, axes = plt.subplots(n_imfs, 2, figsize=(15, 3 * n_imfs), sharex='col')
    
        fig.suptitle(f'Décomposition EMD et spectre - {column}', fontsize=16)
    
        for i, imf in enumerate(imfs):
            # Courbe IMF
            axes[i, 0].plot(TG_d.index, imf, label=f'IMF {i+1}')
            axes[i, 0].set_title(f'IMF {i+1} ({explained_variances[i]*100:.2f}% de la variance)')
            axes[i, 0].legend(loc='upper right')
    
            # Spectre en fréquence
            n = len(imf)
            dt = (TG_d.index[1] - TG_d.index[0]).total_seconds()
            freqs = fftfreq(n, dt)
            fft_vals = np.abs(fft(imf))
            
            # On ne garde que les fréquences positives
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            fft_vals = fft_vals[pos_mask]
    
            # Conversion en cycles/an
            freqs_in_year = freqs * 31536000
            
            # Filtrage jusqu'à 5 cycles/an, par exemple
            freq_mask = freqs_in_year <= 5
            freqs_filtered = freqs_in_year[freq_mask]
            fft_vals_filtered = fft_vals[freq_mask]
            
            # Tracé en barres
            axes[i, 1].bar(freqs_filtered, fft_vals_filtered, width=freqs_filtered[1] - freqs_filtered[0], color='blue', alpha=0.7)
            axes[i, 1].set_title(f'Spectre de IMF {i+1}')
            axes[i, 1].set_xlabel('Fréquence (cycles/an)')
            
        axes[0, 0].set_ylabel('Amplitude')
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # pour laisser la place au titre
        #plt.savefig(os.path.join(save_path, f'TG_{column}_EMD_spectrum.png'), dpi=400)
        plt.show()
    
 #--Seanoe--
for column in S_d.columns:
    if column==22:
        data = (S_d[column].to_numpy())*1e3
        
        # Appliquer EMD
        emd = EMD()
        imfs = emd(data)
        n_imfs = len(imfs)  
        
        # Calcul de la variance expliquée
        total_variance = np.var(data)
        explained_variances = [np.var(imf) / total_variance for imf in imfs]
        
        add_sum = n_imfs >= 6
        
        # Ajuste le nombre de lignes si on ajoute IMF5+IMF6
        n_rows = n_imfs + 1 if add_sum else n_imfs
        
        # Préparer la figure avec 2 colonnes : IMF et spectre
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3 * n_rows), sharex='col')
        fig.suptitle(f'Décomposition EMD et spectre - {column}', fontsize=16)
        
        # Boucle normale sur les IMF
        for i, imf in enumerate(imfs):
            # Courbe IMF
            axes[i, 0].plot(S_d.index, imf, label=f'IMF {i+1}')
            axes[i, 0].set_title(f'IMF {i+1} ({explained_variances[i]*100:.2f}% de la variance)')
            axes[i, 0].legend(loc='upper right')
        
            # Spectre en fréquence
            n = len(imf)
            dt = (S_d.index[1] - S_d.index[0]).total_seconds()
            freqs = fftfreq(n, dt)
            fft_vals = np.abs(fft(imf))
        
            # Fréquences positives
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            fft_vals = fft_vals[pos_mask]
            
            # Conversion en cycles/an
            freqs_in_year = freqs * 31536000
            
            # Filtrage jusqu'à 5 cycles/an, par exemple
            freq_mask = freqs_in_year <= 5
            freqs_filtered = freqs_in_year[freq_mask]
            fft_vals_filtered = fft_vals[freq_mask]
            
            axes[i, 1].bar(freqs_filtered,  fft_vals_filtered, width=freqs_filtered[1] - freqs_filtered[0], color='blue', alpha=0.7)
            axes[i, 1].set_title(f'Spectre de IMF {i+1}')
            axes[i, 1].set_xlabel('Fréquence (cycles/an)')
        
        # Ajustements
        axes[0, 0].set_ylabel('Amplitude')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        #plt.savefig(os.path.join(save_path, f'SEANOE_{column}_EMD_spectrum.png'), dpi=400)
        plt.show()

 #--supperposition--
for col_combined, col_seanoe in zip(TG_d.columns,S_d.columns):
    data1 = TG_d[col_combined].to_numpy()
    data1 = data1 - np.mean(data1)
    data2 = (S_d[col_seanoe].to_numpy()) * 1e3  # pour mise à l’échelle

    # EMD
    emd1 = EMD()
    imfs1 = emd1(data1)
    
    emd2 = EMD()
    imfs2 = emd2(data2)
    n_imfs = len(imfs2 )

    fig, axes = plt.subplots(n_imfs, 1, figsize=(15, 3 * n_imfs))
    fig.suptitle(f'Décomposition EMD et spectre - {col_combined}', fontsize=16)
    for i in range(n_imfs):
        # IMF Seanoe (violet) et Observée (bleu)
        axes[i].plot(S_d.index, imfs2[i], label='Seanoe', color='purple')
        axes[i].plot(TG_d.index, imfs1[i], label='Observé', color='blue')
        axes[i].set_title(f'IMF {i+1}')
        axes[i].legend()

    axes[0].set_ylabel('Amplitude')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    #plt.savefig(os.path.join(save_path, f'supperposition_{col_combined}_EMD_spectrum.png'), dpi=400)
    plt.show()
    
 #--TG all--
mask_70= (df_combined_all.index.year >= 1970) & (df_combined_all.index.year <= 2021)
df_combined_70 = df_combined_all.loc[mask_70]

mask_50= (df_combined_all.index.year >= 1950) & (df_combined_all.index.year <= 2021)
df_combined_50 = df_combined_all.loc[mask_50]

for column in df_combined_70.columns:
    data = df_combined_70[column].dropna()
    time=data.index
    data = data.to_numpy()
    data = data - np.mean(data)

    # Appliquer EMD
    emd = EMD()
    imfs = emd(data)
    n_imfs = len(imfs)

    # Calcul de la variance expliquée
    total_variance = np.var(data)
    explained_variances = [np.var(imf) / total_variance for imf in imfs]

    # Préparer la figure avec 2 colonnes : IMF et spectre
    fig, axes = plt.subplots(n_imfs, 2, figsize=(15, 3 * n_imfs), sharex='col')

    fig.suptitle(f'Décomposition EMD et spectre - {column}', fontsize=16)

    for i, imf in enumerate(imfs):
        # Courbe IMF
        axes[i, 0].plot(time, imf, label=f'IMF {i+1}')
        axes[i, 0].set_title(f'IMF {i+1} ({explained_variances[i]*100:.2f}% de la variance)')
        axes[i, 0].legend(loc='upper right')

        # Spectre en fréquence
        n = len(imf)
        dt = (time[1] - time[0]).total_seconds()
        freqs = fftfreq(n, dt)
        fft_vals = np.abs(fft(imf))
        
        # On ne garde que les fréquences positives
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]
        
        # Conversion en cycles/an
        freqs_in_year = freqs * 31536000
        
        # Filtrage jusqu'à 5 cycles/an, par exemple
        freq_mask = freqs_in_year <= 5
        freqs_filtered = freqs_in_year[freq_mask]
        fft_vals_filtered = fft_vals[freq_mask]
        
        axes[i, 1].bar(freqs_filtered,  fft_vals_filtered, width=freqs_filtered[1] - freqs_filtered[0], color='blue', alpha=0.7)
        axes[i, 1].set_title(f'Spectre de IMF {i+1}')
        axes[i, 1].set_xlabel('Fréquence (cycles/an)')

    axes[0, 0].set_ylabel('Amplitude')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # pour laisser la place au titre
    plt.savefig(os.path.join(save_path, f'TG_{column}_EMD_spectrum_70.png'), dpi=400)
    plt.show()
    

#%%

def plot_emd_and_spectrum(tg_station: str, seanoe_station: int, TG_d=None, S_d=None, save=False, folder=None):
    """
    Visualise les IMF et leurs spectres pour une station marégraphique et une station SEANOE.
    Retourne les IMF calculées.
    
    Returns:
    - imfs_TG : liste des IMF pour la station TG (ou None si station absente)
    - imfs_S : liste des IMF pour la station SEANOE (ou None si station absente)
    """

    imfs_TG = None
    imfs_S = None

    # ---------------- TG ----------------
    if tg_station in TG_d.columns:
        print(f"📡 Traitement TG : {tg_station}")
        data = TG_d[tg_station].to_numpy()
        data = data - np.mean(data)

        emd = EMD()
        imfs_TG = emd(data)
        n_imfs = len(imfs_TG)

        total_variance = np.var(data)
        explained_variances = [np.var(imf) / total_variance for imf in imfs_TG]
        
        fig, axes = plt.subplots(n_imfs, 2, figsize=(15, 3 * n_imfs), sharex='col')
        fig.suptitle(f'Décomposition EMD et spectre - {tg_station} (TG)', fontsize=16)
        
        for i, imf in enumerate(imfs_TG):
            # IMF
            axes[i, 0].plot(TG_d.index, imf, label=f'IMF {i+1}')
            axes[i, 0].set_title(f'IMF {i+1} ({explained_variances[i]*100:.2f}% de la variance)')
            axes[i, 0].legend(loc='upper right')
            
            # Spectre
            n = len(imf)
            dt = (TG_d.index[1] - TG_d.index[0]).total_seconds()
            freqs = fftfreq(n, dt)
            fft_vals = np.abs(fft(imf))
            
            freqs = freqs[freqs > 0]
            fft_vals = fft_vals[:len(freqs)]
            
            freqs_in_year = freqs * 31536000
            freq_mask = freqs_in_year <= 5
            freqs_filtered = freqs_in_year[freq_mask]
            fft_vals_filtered = fft_vals[freq_mask]
            
            axes[i, 1].bar(freqs_filtered, fft_vals_filtered, width=freqs_filtered[1] - freqs_filtered[0], color='blue', alpha=0.7)
            axes[i, 1].set_title(f'Spectre de IMF {i+1}')
            axes[i, 1].set_xlabel('Fréquence (cycles/an)')
        
        axes[0, 0].set_ylabel('Amplitude')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        if save and folder:
            plt.savefig(os.path.join(folder, f"EMD_and_spectrum_{tg_station}.png"), dpi=300)
        else:
            plt.show()
            plt.close()
    else:
        print(f"❌ Station TG '{tg_station}' non trouvée.")

    # ---------------- SEANOE ----------------
    if seanoe_station in S_d.columns:
        print(f"🌊 Traitement SEANOE : {seanoe_station}")
        data = S_d[seanoe_station].to_numpy() * 1e3

        emd = EMD()
        imfs_S = emd(data)
        n_imfs = len(imfs_S)
        
        total_variance = np.var(data)
        explained_variances = [np.var(imf) / total_variance for imf in imfs_S]
        
        add_sum = n_imfs >= 6
        n_rows = n_imfs + 1 if add_sum else n_imfs
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3 * n_rows), sharex='col')
        fig.suptitle(f'Décomposition EMD et spectre - {seanoe_station} (SEANOE)', fontsize=16)
        
        for i, imf in enumerate(imfs_S):
            axes[i, 0].plot(S_d.index, imf, label=f'IMF {i+1}')
            axes[i, 0].set_title(f'IMF {i+1} ({explained_variances[i]*100:.2f}% de la variance)')
            axes[i, 0].legend(loc='upper right')
            
            n = len(imf)
            dt = (S_d.index[1] - S_d.index[0]).total_seconds()
            freqs = fftfreq(n, dt)
            fft_vals = np.abs(fft(imf))
            
            freqs = freqs[freqs > 0]
            fft_vals = fft_vals[:len(freqs)]
            
            freqs_in_year = freqs * 31536000
            freq_mask = freqs_in_year <= 5
            freqs_filtered = freqs_in_year[freq_mask]
            fft_vals_filtered = fft_vals[freq_mask]
            
            axes[i, 1].bar(freqs_filtered, fft_vals_filtered, width=freqs_filtered[1] - freqs_filtered[0], color='blue', alpha=0.7)
            axes[i, 1].set_title(f'Spectre de IMF {i+1}')
            axes[i, 1].set_xlabel('Fréquence (cycles/an)')
        
        axes[0, 0].set_ylabel('Amplitude')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        if save and folder:
            plt.savefig(os.path.join(folder, f"EMD_and_spectrum_{seanoe_station}.png"), dpi=300)
        else:
            plt.show()
            plt.close()
    else:
        print(f"❌ Station SEANOE '{seanoe_station}' non trouvée.")

    return imfs_TG, imfs_S

def plot_merged_imf_combo_correlation(imfs_TG, TG_combos, imfs_S, Seanoe_combos):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Fusion des IMF simples et des combinaisons
    all_TG = imfs_TG + [combo[1] for combo in TG_combos]
    all_S = imfs_S + [combo[1] for combo in Seanoe_combos]

    TG_labels = [f"TG IMF {i+1}" for i in range(len(imfs_TG))] + [f"TG {c[0]}" for c in TG_combos]
    S_labels = [f"Seanoe IMF {i+1}" for i in range(len(imfs_S))] + [f"Seanoe {c[0]}" for c in Seanoe_combos]

    # Matrice de corrélation croisée
    corr_matrix = np.array([
        [np.corrcoef(tg, s)[0, 1] for s in all_S]
        for tg in all_TG
    ])

    # Création du DataFrame
    df_corr = pd.DataFrame(corr_matrix, index=TG_labels, columns=S_labels)

    # Affichage de la heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Corrélation entre IMF + Combos : TG vs SEANOE", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def generate_combinations(imfs, max_comb=3):
    """
    Génère les combinaisons additives d'IMF jusqu'à max_comb IMF à la fois.
    """
    combos = []
    for r in range(1, max_comb + 1):
        for idxs in combinations(range(len(imfs)), r):
            summed = np.sum([imfs[i] for i in idxs], axis=0)
            combos.append((idxs, summed))
    return combos

def plot_highly_correlated_pairs(imfs_TG, TG_combos, imfs_S, Seanoe_combos, threshold=0.7):

    # Fusion des IMF simples et des combinaisons
    all_TG = imfs_TG + [combo[1] for combo in TG_combos]
    all_S = imfs_S + [combo[1] for combo in Seanoe_combos]

    TG_labels = [f"TG IMF {i+1}" for i in range(len(imfs_TG))] + [f"TG {c[0]}" for c in TG_combos]
    S_labels = [f"Seanoe IMF {i+1}" for i in range(len(imfs_S))] + [f"Seanoe {c[0]}" for c in Seanoe_combos]

    # Recherche des paires avec corrélation >= threshold
    for i, tg_label in enumerate(TG_labels):
        for j, s_label in enumerate(S_labels):
            corr = np.corrcoef(all_TG[i], all_S[j])[0, 1]
            if threshold <= corr <= 1:
                tg_norm = (all_TG[i] - np.mean(all_TG[i])) / np.std(all_TG[i])
                s_norm = (all_S[j] - np.mean(all_S[j])) / np.std(all_S[j])

                plt.figure(figsize=(10, 5))
                plt.plot(tg_norm, label=tg_label, linestyle='-')
                plt.plot(s_norm, label=s_label, linestyle='--')
                plt.title(f"{tg_label} vs {s_label} — Corrélation = {corr:.2f}")
                plt.xlabel("Index")
                plt.ylabel("Valeurs normalisées")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

#%%

imfs_TG, imfs_S = plot_emd_and_spectrum("GALVESTON II", 12, TG_d=TG_d, S_d=S_d)

# Assure-toi que les IMF ont la même longueur
min_len = min(len(imfs_TG[0]), len(imfs_S[0]))
imfs_TG = [imf[:min_len] for imf in imfs_TG]
imfs_S = [imf[:min_len] for imf in imfs_S]

# Créer une matrice de corrélation
corr_matrix = np.zeros((len(imfs_TG), len(imfs_S)))

for i, tg_imf in enumerate(imfs_TG):
    for j, seanoe_imf in enumerate(imfs_S):
        corr = np.corrcoef(tg_imf, seanoe_imf)[0, 1]
        corr_matrix[i, j] = corr

# Transformer en DataFrame pour une jolie heatmap
corr_df = pd.DataFrame(
    corr_matrix,
    index=[f"TG IMF {i+1}" for i in range(len(imfs_TG))],
    columns=[f"Seanoe IMF {j+1}" for j in range(len(imfs_S))]
)

# Affichage
plt.figure(figsize=(10, 6))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("🧩 Corrélation entre les IMF - PENSACOLA")
plt.tight_layout()
plt.show()

from itertools import combinations


# S'assurer que les IMF sont de même longueur
min_len = min(len(imfs_TG[0]), len(imfs_S[0]))
imfs_TG = [imf[:min_len] for imf in imfs_TG]
imfs_S = [imf[:min_len] for imf in imfs_S]

# Générer les combinaisons (jusqu'à 3 IMF à la fois)
TG_combos = generate_combinations(imfs_TG, max_comb=3)
Seanoe_combos = generate_combinations(imfs_S, max_comb=3)

plot_merged_imf_combo_correlation(imfs_TG, TG_combos, imfs_S, Seanoe_combos)
plot_highly_correlated_pairs(imfs_TG, TG_combos, imfs_S, Seanoe_combos, threshold=0.8)

#%%

def analyse_emd_correlations_all_pairs(TG_d, S_d, mapping, threshold=0.8, max_comb=5, output_dir="emd_analysis"):
    """
    Applique l'analyse EMD + spectre + corrélations à toutes les paires de stations données.
    Sauvegarde tous les graphiques dans un dossier par station.
    
    mapping : dict, clé = nom dans TG_d, valeur = identifiant colonne dans S_d
    """
    os.makedirs(output_dir, exist_ok=True)

    for tg_station, seanoe_id in mapping.items():
        print(f"\n================= 📍 Analyse de {tg_station} ↔ SEANOE {seanoe_id} =================")
        station_dir = os.path.join(output_dir, f"{tg_station.replace(' ', '_')}_{seanoe_id}")
        os.makedirs(station_dir, exist_ok=True)

        try:
            # EMD + spectres
            imfs_TG, imfs_S = plot_emd_and_spectrum(
                tg_station, seanoe_id, TG_d=TG_d, S_d=S_d, save=True, folder=station_dir
            )

            if imfs_TG is None or imfs_S is None:
                print("⛔ Données manquantes, on passe à la suivante.")
                continue

            # Troncature pour égaliser les longueurs
            min_len = min(len(imfs_TG[0]), len(imfs_S[0]))
            imfs_TG = [imf[:min_len] for imf in imfs_TG]
            imfs_S = [imf[:min_len] for imf in imfs_S]

            # Combinaisons IMF
            TG_combos = generate_combinations(imfs_TG, max_comb=max_comb)
            Seanoe_combos = generate_combinations(imfs_S, max_comb=max_comb)

            # Matrice de corrélation
            plot_merged_imf_combo_correlation(
                imfs_TG, TG_combos, imfs_S, Seanoe_combos, save=True, folder=station_dir
            )

            # Tracer les paires corrélées
            plot_highly_correlated_pairs(
                imfs_TG, TG_combos, imfs_S, Seanoe_combos, threshold=threshold, save=True, folder=station_dir
            )

        except Exception as e:
            print(f"⚠️ Erreur pendant l’analyse de {tg_station} ↔ {seanoe_id} : {e}")



mapping = {
    'KEY WEST':29,
    'NAPLES': 28,
    'FORT MYERS': 27,
    'ST. PETERSBURG': 26,
    'CEDAR KEY II': 25,
    'APALACHICOLA': 23,
    'PENSACOLA': 22,
    'DAUPHIN ISLAND': 21,
    'GRAND ISLE': 15,
    'GALVESTON II': 12,
    'FREEPORT': 11,
    'ROCKPORT': 10
}

analyse_emd_correlations_all_pairs(TG_d, S_d, mapping)
