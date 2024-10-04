from DZ import *

# SVM klasa
class SVM:
    def __init__(self, C=None, kernel=None, degree=3):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.w = None
        self.b = None
        self.alphas = None

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def polynomial_kernel(self, X1, X2, degree=3):
        return (1 + np.dot(X1, X2.T)) ** degree

    def fit(self, X, y):
        m, n = X.shape
        y = y.astype(float)

        if self.kernel is None:
            self.kernel = self.linear_kernel  # Default to linear kernel if not specified

        # Izračunavanje kernel matrice
        K = self.kernel(X, X)

        # Konstruisanje H matrice za kvadratno programiranje
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
        self.alphas = np.ravel(solution['x'])

        # Vektor težina
        self.w = ((self.alphas * y).T @ X).reshape(-1, 1)

        # Određivanje pristrasnosti (bias)
        S = (self.alphas > 1e-4)
        # Množenje matrica i računanje pristrasnosti
        self.b = np.mean(y[S] - np.dot(K[S][:, S], self.alphas[S] * y[S]))

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("Model not trained yet.")
        return np.sign(np.dot(self.kernel(X, X_train), self.alphas * y_train) + self.b).flatten()

# Treniranje SVM modela
svm = SVM(C=1.0, kernel=SVM.polynomial_kernel, degree=3)
svm.fit(X_train, y_train)

# Predikcija
y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)

# Prikaz rezultata
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='r', label='Class -1')

# Separaciona prava
x_plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y_plot = -(svm.w[0] * x_plot + svm.b) / svm.w[1]
plt.plot(x_plot, y_plot, color='k', label='Decision Boundary')

# Prikaz nosećih vektora
support_vectors_idx = np.where((svm.alphas > 1e-4))[0]
plt.scatter(X[support_vectors_idx][:, 0], X[support_vectors_idx][:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

plt.legend()
plt.show()

# Prikaz tačnosti na obučavajućem i test skupu
accuracy_train = np.mean(y_pred_train == y_train)
print("Train Accuracy: {:.2f}%".format(accuracy_train * 100))
accuracy_test = np.mean(y_pred_test == y_test)
print("Test Accuracy: {:.2f}%".format(accuracy_test * 100))
