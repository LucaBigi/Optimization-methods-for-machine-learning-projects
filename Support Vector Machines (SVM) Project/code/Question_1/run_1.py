from functions_1 import *



cwd = os.getcwd()
X_train, X_test, Y_train, Y_test = load_dataset(cwd, kind='train')

##GRID SEARCH
#minimo_validation = grid_search_with_k_fold_cross_validation(X_train, Y_train)
#print("[mean_training_error_rate", "mean_validation_error_rate", "gamma", "C] = ",minimo_validation ) #inoltre la funzione grid_search_with_k_fold_cross_validation salva un file excel con i risultati della grid search
#Grafici_grid_search("grid_search.xlsx")

#SCELTA FINALE IPERPARAMETRI
gamma=0.00008 ###NB:     KERNEL RBF --> gamma > 0     |     KERNEL POLINOMIALE --> gamma >= 1 
C=15

##SCEGLI IL KERNEL (abbiamo scelto di usare il kernel rbf):
K_train=rbf_kernel(X_train,X_train, gamma) # kernel rbf ---> VA SCELTO GAMMA > 0
#K_train=polynomial_kernel(X_train,X_train, gamma) ## kernel polinomiale ---> VA SCELTO GAMMA > 1

#risolvo il duale
start_time = time.time()
alpha, n_iter, KKT_viol, solver_status = solve_svm_dual(K_train, Y_train, C) 
end_time = time.time()
optimization_time=end_time-start_time

#SCEGLI IL KERNEL COERENTEMENTE A QUELLO SCELTO PRIMA:
K_test=rbf_kernel(X_train,X_test, gamma) # kernel rbf ---> VA SCELTO GAMMA > 0
#K_test=polynomial_kernel(X_train,X_test, gamma) # kernel polinomiale ---> NB: VA SCELTO GAMMA > 1

y_train=predict(Y_train,K_train, K_train, alpha, C) 
y_test=predict(Y_train,K_train, K_test, alpha, C)
res_train=y_train*Y_train
res_test=y_test*Y_test
error_rate_train=np.count_nonzero(res_train == -1)/len(res_train)
error_rate_test=np.count_nonzero(res_test == -1)/len(res_test)
optimal_f=dual_objective(K_train, Y_train, alpha)



#CREAZIONE TABELLA OUTPUT
tabella=    [["1. C", C], ["2. Gamma", gamma],["3. Accuracy on training set", 1-error_rate_train], ["4. Accuracy on test set", 1-error_rate_test], ["5. Optimization Time (seconds)", optimization_time],
             ["6. Iterations",n_iter],["7. KKT violations (m-M)", KKT_viol], ["8. Optimal f value",optimal_f], ["9. Solver status", solver_status]]
                            
# Larghezza colonne e spaziatura manuale
col1_width = 40  # Larghezza fissa per la prima colonna
col2_width = 10  # Larghezza fissa per la seconda colonna
spaziatura = 5   # Spaziatura tra le colonne

# Stampa dell'intestazione
print(f"{''.ljust(col1_width)}{' ' * spaziatura}{'Q1 Values'.ljust(col2_width)}")

# Stampa delle righe della tabella
for riga in tabella:
    print(f"{str(riga[0]).ljust(col1_width)}{' ' * spaziatura}{str(riga[1]).ljust(col2_width)}")


print_confusion_matrix(y_test, Y_test)