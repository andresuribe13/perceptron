import random
import numpy as np
from numpy import genfromtxt
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # libreria de reduccion de dimensionalidad
style.use('ggplot')

DataBase = genfromtxt('dataset.txt', delimiter=',')

X=(np.array([DataBase[:,0],DataBase[:,1],DataBase[:,2],DataBase[:,3],DataBase[:,4],DataBase[:,5],DataBase[:,6]]).T)
H = np.array([DataBase[:,7]])

x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2

X = PCA(n_components=2).fit_transform(X) # para pasar de 7 caracteristicas a 2

plt.scatter(X[:, 0], X[:, 1], c='black', label='Datos no agrupados')
plt.title('Kmeans')
plt.show()

m = X.shape[0] # Cantidad de muestras en la base de datos  --FILAS
n = X.shape[1] # Cantidad de caracteristicas en la base de datos --COLUMNAS

nIteraciones = 500
k = 2 # Cantidad de centroides

# Creacion de los centroides o centros

centroides = np.array([]).reshape(n, 0) #crea arreglo vacio de n filas y una columna
for i in range(k):
    rand = random.randint(0, m-1) #genera un numero aletorio para elegir una de las columnas de las entradas
    centroides = np.c_[centroides, X[rand]]
    print(centroides)
    
output = {}

distanciaEuclidiana = np.array([]).reshape(m, 0)

#calcula las distancias para los centros
for j in range(k):
    tempDist = np.sum(abs(X - centroides[:, j]), axis=1) 
    #print(tempDist)
    distanciaEuclidiana = np.c_[distanciaEuclidiana, tempDist] #(distanciaEuclidiana)
    
distanciaMinima = np.argmin(distanciaEuclidiana, axis=1) + 1

Y = {}

# Asignamos los puntos/datos a un cluster con base a la disancia mìnima calculada

for l in range(k):
    Y[l+1] = np.array([]).reshape(2, 0)
    
for p in range(m):
    Y[distanciaMinima[p]] = np.c_[Y[distanciaMinima[p]], X[p]] #np.c_ sirve para concatenar

for q in range(k):
    Y[q+1] = Y[q+1].T
    
for r in range(k):
    centroides[:, r] = np.mean(Y[r+1], axis=0)

#se calcula las distancia para todos los datos 
for i in range(nIteraciones):
    distanciaEuclidiana = np.array([]).reshape(m, 0)
    for j in range(k):

        tempDist = np.sum(abs(X - centroides[:, j]), axis=1)
        distanciaEuclidiana = np.c_[distanciaEuclidiana, tempDist]
    
    distanciaMinima = np.argmin(distanciaEuclidiana, axis=1) + 1
    #print(distanciaMinima)

    Y = {}

    for l in range(k):
        Y[l+1] = np.array([]).reshape(2, 0)
        
    for p in range(m):
        Y[distanciaMinima[p]] = np.c_[Y[distanciaMinima[p]], X[p]] #concatena las salidad de las etiquetas y las entradas

    for q in range(k):
        Y[q+1] = Y[q+1].T
        
    for r in range(k):
        centroides[:, r] = np.mean(Y[r+1], axis=0)
    output = Y

color = ['red','blue','green']
labels = ['cluster1','cluster2','cluster3']

for s in range(k):
    plt.scatter(output[s+1][:, 0], output[s+1][:, 1], c=color[s], label=labels[s])

distancia = np.array([distanciaMinima])

ac = 0
ma = 0

for i in range(720):
    H[0,i] = H[0,i] + 1
    print ("{};{}".format(int(H[0,i]),distancia[0,i]))
    
    if int(H[0,i]) == int(distancia[0,i]):
         ma = ma +1
    else:
         a = ac +1
         
porerror= (ma/720)*100
print("Buenos",ac)
print("malos",ma)
print("Procentaje de error",porerror)
plt.scatter(centroides[0, :], centroides[1, :], s=100, c='yellow', label='centroides')
plt.show()