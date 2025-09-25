from functions_2 import *

cwd = os.getcwd()
X_train, X_test, Y_train, Y_test = load_dataset(cwd, kind='train')

#grafici per scelta q:
#gamma=0.00008 ###NB:     KERNEL RBF --> gamma > 0     |     KERNEL POLINOMIALE --> gamma >= 1 
#C=15
#error_rate_list=[]
#optimization_time_list=[]
#n_subproblems_list=[]
#KKT_viol_list=[]
#for q in range (4,90,2):
#    print("q = ",q)
#    start_time = time.time()
#    alpha, external_iter, tot_iter, KKT_viol, last_solver_status = decomposition_algorithm(X_train, Y_train, q, gamma, C) #external_iter rappresenta il numero di sottoproblemi risolti prima di soddisfare le KKT
#    end_time = time.time()
#    K_test=rbf_kernel(X_train,X_test, gamma) # kernel rbf ---> VA SCELTO GAMMA > 0
#    K_train=rbf_kernel(X_train,X_train, gamma) 
#    y_test=predict(Y_train,K_train, K_test, alpha, C)
#    res_test=y_test*Y_test
#    error_rate_test=np.count_nonzero(res_test == -1)/len(res_test)
#    optimization_time_list.append((end_time-start_time))
#    error_rate_list.append(error_rate_test)
#    n_subproblems_list.append(external_iter)
#    KKT_viol_list.append(KKT_viol)
#plot_q_variation(error_rate_list, optimization_time_list, n_subproblems_list, KKT_viol_list)
########################################################


#SCELTA IPERPARAMETRI
gamma=0.00008 ###NB:     KERNEL RBF --> gamma > 0     |     KERNEL POLINOMIALE --> gamma >= 1 
C=15
q=90

start_time = time.time()
alpha, _, tot_iter, KKT_viol, solver_status = decomposition_algorithm(X_train, Y_train, q, gamma, C) #external_iter rappresenta il numero di sottoproblemi risolti prima di soddisfare le KKT
end_time = time.time()

K_test=rbf_kernel(X_train,X_test, gamma) # kernel rbf ---> VA SCELTO GAMMA > 0





K_train=rbf_kernel(X_train,X_train, gamma) 
y_train=predict(Y_train,K_train, K_train, alpha, C) 
y_test=predict(Y_train,K_train, K_test, alpha, C)
res_train=y_train*Y_train
res_test=y_test*Y_test
error_rate_train=np.count_nonzero(res_train == -1)/len(res_train)
error_rate_test=np.count_nonzero(res_test == -1)/len(res_test)
optimal_f=dual_objective(K_train, Y_train, alpha)



#CREAZIONE TABELLA OUTPUT
tabella=    [["1. C", C], ["2. Gamma", gamma], ["3. q", q], ["4. Accuracy on training set", 1-error_rate_train], ["5. Accuracy on test set", 1-error_rate_test], ["6. Run Time (seconds)", end_time-start_time],
             ["7. Total number iterations",tot_iter], ["8. KKT violations (m-M)", KKT_viol], ["9. Optimal f value",optimal_f], ["10. Distinct Solver State History", solver_status]]
                            
# Larghezza colonne e spaziatura manuale
col1_width = 40  # Larghezza fissa per la prima colonna
col2_width = 10  # Larghezza fissa per la seconda colonna
spaziatura = 5   # Spaziatura tra le colonne

# Stampa dell'intestazione
print(f"{''.ljust(col1_width)}{' ' * spaziatura}{'Q2 Values'.ljust(col2_width)}")

# Stampa delle righe della tabella
for riga in tabella:
    print(f"{str(riga[0]).ljust(col1_width)}{' ' * spaziatura}{str(riga[1]).ljust(col2_width)}")

print_confusion_matrix(y_test, Y_test)