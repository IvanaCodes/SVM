# Biblioteke - nije ok
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

# Učitavanje podataka
data = pd.read_csv("svmData 3.csv", header=None)
print(data)

# Broj primera i odlika
n = data.shape[1] - 1
print("Broj odlika:", n)
m = data.shape[0]
print("Broj primera:", m)

# Matrica prediktora
X = data.iloc[:, 0:2].values
print("Prikaz prediktora\n", X)
print("Dimenzije matrice prediktora:", X.shape)

# Ciljna promenljiva
y = data.iloc[:, -1].values
print("Ciljna promenljiva\n", y)
print("Dimenzije vektora y:", y.shape)

# Transformacija oznaka klasa na {-1, 1}
y = y * 2 - 1

# Dodavanje kolone sa jedinicama za računanje w i b zajedno
X = np.hstack((X, np.ones((X.shape[0], 1))))

# Prikaz podataka
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='r', label='Class -1')
plt.legend()
plt.show()

# Podela obučavajućeg skupa na test i training skupove
split_index = int(0.7 * len(data))

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Ispis train i test promenljivih
print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)

# SVM klasa
class SVM:
    def __init__(self, C=None):
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape
        y = y.astype(float)

        # Konstruisanje H matrice za kvadratno programiranje
        K = np.dot(X, X.T)
        H = np.outer(y, y) * K
        P = cvxopt_matrix(H)

        q = cvxopt_matrix(-np.ones((m, 1)))

        if self.C is None:
            # Tvrda margina - Hard margin SVM
            G = cvxopt_matrix(-np.eye(m))
            h = cvxopt_matrix(np.zeros(m))
        else:
            # Meka margina sa regularizacijom
            G_std = np.eye(m) * -1
            G_slack = np.eye(m)
            G = cvxopt_matrix(np.vstack((G_std, G_slack)))
            h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))

        A = cvxopt_matrix(y, (1, m), 'd')
        b = cvxopt_matrix(0.0)

        # Resenje kvadratnog problema
        cvxopt_solvers.options['show_progress'] = False
        solution = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Vektor težina
        self.w = ((alphas * y).T @ X).reshape(-1, 1)

        # Indeksi nosećih vektora
        S = (alphas > 1e-4)
        self.b = np.mean(y[S] - np.dot(X[S], self.w))

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b).flatten()

# Treniranje SVM modela
svm = SVM(C=1.0)
svm.fit(X_train, y_train)

# Predikcija
y_pred = svm.predict(X_test)

# Prikaz rezultata
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='r', label='Class -1')

# Separaciona prava
x_plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y_plot = -(svm.w[0] * x_plot + svm.b) / svm.w[1]
plt.plot(x_plot, y_plot, color='k', label='Decision Boundary')

# Prikaz nosećih vektora
support_vectors_idx = np.where((alphas > 1e-4))[0]
plt.scatter(X[support_vectors_idx][:, 0], X[support_vectors_idx][:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

plt.legend()
plt.show()

# Prikaz tačnosti na test skupu
accuracy = np.mean(y_pred == y_test)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
