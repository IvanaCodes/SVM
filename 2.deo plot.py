from DD import *
import matplotlib.pyplot as plt

# Generisanje različitih vrednosti za hiperparametar C
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Liste za čuvanje tačnosti na trening i test skupu
train_accuracies = []
test_accuracies = []

# Iteracija kroz različite vrednosti parametra C
for C_val in C_values:
    # Treniranje SVM modela sa trenutnom vrednošću C
    svm = SVM(C=C_val)
    svm.fit(X_train, y_train)

    # Predikcija na trening i test skupu
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)

    # Računanje tačnosti
    accuracy_train = np.mean(y_pred_train == y_train)
    accuracy_test = np.mean(y_pred_test == y_test)

    # Čuvanje tačnosti
    train_accuracies.append(accuracy_train)
    test_accuracies.append(accuracy_test)

# Prikazivanje rezultata
plt.plot(C_values, train_accuracies, label='Train Accuracy')
plt.plot(C_values, test_accuracies, label='Test Accuracy')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Effect of C on Accuracy')
plt.legend()
plt.show()
