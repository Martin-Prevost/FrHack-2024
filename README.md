# FRHACK2024 : hackaton des fréquences

*Organisé par l'*[ANFR](https://www.anfr.fr)

**Qu'est ce que c'est ?**

Ce script est un outil créé pendant le hackaton des fréquences organisé par l'ANFR.
Il permet de générer un carroyage avec 3 types de zones (urbain, peri-urbain et rurale) à partir de fichiers shapefiles de départements.

**Comment utiliser ce script ?**

Ce script est un outil en ligne de commande. Tous les paramètres de générations des shapefiles peuvent être réglés avec les différentes options :
```
$ python main.py --help
usage: main.py [-h] [--filename FILENAME] [--peri-urbaines_file PERI_URBAINES_FILE] [--rurales-file RURALES_FILE] [--urbaines-file URBAINES_FILE] [--size-urb SIZE_URB] [--selected_operator SELECTED_OPERATOR]
               [--selected_techno {4G,5G,all}] [--predict PREDICT] [--nb_val_moy NB_VAL_MOY]

Challenge 3 command line tool

options:
  -h, --help            show this help message and exit
  --filename FILENAME   the name of the data file (default: "data/Mesures sur 41 45 89.csv")
  --peri-urbaines_file PERI_URBAINES_FILE
                        the name of the peri-urban shape file (default: "data/Zones PERI URBAINES 41 45 89.shp")
  --rurales-file RURALES_FILE
                        the name of the rural shape file (default: "data/Zones RURALES 41 45 89.shp")
  --urbaines-file URBAINES_FILE
                        the name of the urban shape file (default: "data/Zones URBAINES 41 45 89.shp")
  --size-urb SIZE_URB   the size of the urban area (default: 1500)
  --selected_operator SELECTED_OPERATOR
                        the selected operator (default: "OP1")
  --selected_techno {4G,5G,all}
                        the selected technology (default: "all", choices: ["4G", "5G", "all"])
  --predict PREDICT     Add the prediction
  --nb_val_moy NB_VAL_MOY
                        Number of values used for average calculation
```

Les fichiers shapefiles seront générés dans le dossier `output/`
Pour que le script tourne automatiquement, les fichers CSV et shapefiles en entrée peuvent être placés dans le dossier `data/`
