# Analyse des variations du niveau de la mer : marégraphes vs. altimétrie (Golfe du Mexique)

## Contexte
Comparaison marégraphes (PSMSL), altimétrie retrackée (X-TRACK 2002-2021) et GNSS pour estimer le **VLM (Vertical Land Motion)** et l'élévation absolue du niveau marin. Stage M2 Géosciences Littoral, LIENSs (La Rochelle Université/CNRS), encadré par M. Karpytchev (2024-2025). [file:59]

**Objectif** : Filtrer tendances, cohérence sources, VLM long terme (1960-2021).

## Données utilisées
- Marégraphes : 12 stations (ex. Key West, Grand Isle ; 1913-2023).
- Altimétrie : X-TRACK retrack (ALES, Jason/ERS/Sentinel-3 ; SLA + corrections).
- GNSS : ULR8 (2002-2025).

## Méthodologie
1. **Filtrage** : EMD/SSA pour tendances interannuelles (vs. saisonnalité).
2. **VLM direct** : Δ tendance TG-alti (RMSE ~1-2 mm/an vs. GNSS).
3. **Reconstruction long terme** : Projection TG sur EOFs alti → extension 1960-2021.
4. **Régionale** : Méthode Bublé (inversion matricielle pour tendance commune).

## Résultats clés
- Corrélations >90% (TG-alti filtrées).
- Accélération élévation mer post-2010 (+~3-8 mm/an).
- VLM : Subsidence forte (ex. Grand Isle -4/-7 mm/an) ; cohérent GNSS.

## Outils & Code
- Python : Pandas/NumPy (filtrage), xarray (EOF), Matplotlib (séries/cartes).
- SQL (bases), QGIS (spatial).
