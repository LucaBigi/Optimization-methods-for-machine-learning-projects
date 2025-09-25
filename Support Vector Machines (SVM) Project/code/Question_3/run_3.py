from functions_3 import *

cwd = os.getcwd()
X_train, X_test, Y_train, Y_test = load_dataset(cwd, kind='train')


#SCELTA IPERPARAMETRI
gamma=0.00008 ###NB:     KERNEL RBF --> gamma > 0   
C=15
start_time = time.time()
alpha, n_of_Q_rows_computed, iterations, KKT_viol = decomposition_algorithm_MVP(X_train, Y_train, gamma, C)
end_time = time.time()
K_test=rbf_kernel(X_train,X_test, gamma)

K_train=rbf_kernel(X_train,X_train, gamma) 
y_train=predict(Y_train,K_train, K_train, alpha, C) 
y_test=predict(Y_train,K_train, K_test, alpha, C)
res_train=y_train*Y_train
res_test=y_test*Y_test
error_rate_train=np.count_nonzero(res_train == -1)/len(res_train)
error_rate_test=np.count_nonzero(res_test == -1)/len(res_test)
optimal_f=dual_objective(K_train, Y_train, alpha)


#CREAZIONE TABELLA OUTPUT
tabella=    [["1. C", C], ["2. Gamma", gamma], ["3. Accuracy on training set", 1-error_rate_train], ["4. Accuracy on test set", 1-error_rate_test], ["5. Run Time (seconds)", end_time-start_time],
             ["6. Number of Q columns computed", n_of_Q_rows_computed],["7. Iterations",iterations], ["8. KKT violations (m-M)", KKT_viol], ["9. Optimal f value",optimal_f]]
                            
# Larghezza colonne e spaziatura manuale
col1_width = 40  # Larghezza fissa per la prima colonna
col2_width = 10  # Larghezza fissa per la seconda colonna
spaziatura = 5   # Spaziatura tra le colonne

# Stampa dell'intestazione
print(f"{''.ljust(col1_width)}{' ' * spaziatura}{'Q3 Values'.ljust(col2_width)}")

# Stampa delle righe della tabella
for riga in tabella:
    print(f"{str(riga[0]).ljust(col1_width)}{' ' * spaziatura}{str(riga[1]).ljust(col2_width)}")

print_confusion_matrix(y_test, Y_test)