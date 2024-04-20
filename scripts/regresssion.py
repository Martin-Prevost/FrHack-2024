import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import pickle

filename = "data/Mesures sur 41 45 89.csv"
dept_file = "data/Shape Depts 41 45 89.shp"
town_file = "data/Shape Blois Orleans Auxerre.shp"
peri_urbaines_file = "data/Zones PERI URBAINES 41 45 89.shp"
rurales_file = "data/Zones RURALES 41 45 89.shp"
urbaines_file = "data/Zones URBAINES 41 45 89.shp"

data = np.genfromtxt(filename, delimiter=';', dtype=str, skip_header=1)

size_urb = 1500
selected_operator = "OP1"
techno_list = ["4G", "5G", "all"]
selected_techno = techno_list[2]

if selected_techno != "all":
    data = data[data[:, 3] == selected_techno]

data = data[data[:, 4] == selected_operator]

ids = data[:, 0]
x = data[:, 1].astype(int)
y = data[:, 2].astype(int)
technos = data[:, 3]
operateurs = data[:, 4]
dbms = data[:, 5].astype(int)
puissances_recues = data[:, 6]

X_train = np.array([x, y]).T
y_train = dbms

print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", y_train.shape)

# Création du modèle k-NN avec noyau gaussien
grilleCV = GridSearchCV(KNeighborsRegressor(), param_grid={"n_neighbors": np.arange(3, 10)}, cv=5)
grilleCV.fit(X_train, y_train)
best_k = grilleCV.best_params_['n_neighbors']
print("Meilleur k:", best_k)
knn_regressor = KNeighborsRegressor(n_neighbors=best_k, metric='euclidean', weights='distance')
knn_regressor.fit(X_train, y_train)

# Generate a mesh grid
x_min, x_max = x.min() - 1, x.max() + 1
y_min, y_max = y.min() - 1, y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1000),
                     np.arange(y_min, y_max, 1000))

# Predict on the mesh grid
Z = knn_regressor.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
print(Z)

# Plot the predicted values
plt.contourf(xx, yy, Z, alpha=0.8)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kernel Ridge Regression')
plt.colorbar()
plt.show()




# Open carroyage.pkl
with open('carroyage.pkl', 'rb') as file:
    carroyage = pickle.load(file)

X0_train_carroyage = []
X1_train_carroyage = []
y_train_carroyage = []

X0_grille_carroyage = []
X1_grille_carroyage = []

for carre in carroyage:
    X0_grille_carroyage.append((carre['s2_gps'][0] - carre['s1_gps'][0])/2 + carre['s1_gps'][0])
    X1_grille_carroyage.append((carre['s4_gps'][1] - carre['s1_gps'][1])/2 + carre['s1_gps'][1])
    if carre['dbm_moy'] != 0.:
        X0_train_carroyage.append((carre['s2_gps'][0] - carre['s1_gps'][0])/2 + carre['s1_gps'][0])
        X1_train_carroyage.append((carre['s4_gps'][1] - carre['s1_gps'][1])/2 + carre['s1_gps'][1])
        y_train_carroyage.append(carre['dbm_moy'])

X_train_carroyage = np.array([X0_train_carroyage, X1_train_carroyage]).T

# Création du modèle k-NN avec noyau gaussien
grilleCV_carroyage = GridSearchCV(KNeighborsRegressor(), param_grid={"n_neighbors": np.arange(1, 10)}, cv=5)
grilleCV_carroyage.fit(X_train_carroyage, y_train_carroyage)
best_k_carroyage = grilleCV_carroyage.best_params_['n_neighbors']
print("Meilleur k:", best_k_carroyage)
knn_regressor_carroyage = KNeighborsRegressor(n_neighbors=best_k_carroyage)
knn_regressor_carroyage.fit(X_train_carroyage, y_train_carroyage)

# Predict on the mesh grid
X_carroyage = np.array([X0_grille_carroyage, X1_grille_carroyage]).T
Z_carroyage = knn_regressor_carroyage.predict(X_carroyage)

for carre in carroyage:
    carre['predict_dbm_moy'] = Z_carroyage[carroyage.index(carre)]






# Plot the predicted values
plt.scatter(X0_grille_carroyage, X1_grille_carroyage, c=Z_carroyage, cmap='RdYlGn')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kernel Ridge Regression')
plt.colorbar()
plt.show()

