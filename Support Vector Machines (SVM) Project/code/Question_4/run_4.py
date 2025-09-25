from functions_4 import *

#SCELTA IPERPARAMETRI
gamma=0.00008###NB:     KERNEL RBF --> gamma > 0     |     KERNEL POLINOMIALE --> gamma >= 1 
C=15

cwd = os.getcwd()
X_train_all, X_test_all, Y_train_all, Y_test_all = load_dataset(cwd, kind='train')
true_label_list=[1,5,7]

#i risultati in termini di funzione obiettivo del duale per 1vs5 vengono diversi dal caso precedente 
#perchè lo split con la randomizzazione iniziale dei dati viene fatta sta volta con un numero superiore
#di dati. infatti se considero solo le etichette 5 e 1 la funzione obiettivo avrà valore analogo a 
#quello che avevamo nei quesiti precedenti
'''#controprova solo per OAO:
true_label_list=[1,5]
#NB:per la controprova elimina anche l'importazione dei dati per il 7 dalla funzione load_data''' 

# ---------------------------------------OAO-----------------------------------------
predict_array=np.array([])
predict_array_train=np.array([])
alpha=np.array([])
error_rate_train_red_list=[None, None, 10000] # metto 1000 alla fine per farlo funzionare anche nel caso della controprova con solo 2 classi
optimization_time=0
KKT_viol_list=[None, None, None]
optimal_f_list=[None, None, None]
iteration_list=[None, None, None]

for i, label in enumerate(true_label_list):
    start_time = time.time()
    set=[true_label_list[i],true_label_list[i-1]]
    X_train_red, Y_train_red = OAO(X_train_all, Y_train_all, set[0], set[1]) #filtro la classe esclusa. La classe in set[1] avrà etichetta in Y_train +1, la classe in set[0] avrà etichetta -1
    alpha_star, _, iterations, KKT_viol, funz =decomposition_algorithm_MVP(X_train_red, Y_train_red, gamma, C)
    end_time = time.time() 
    optimization_time+= (end_time -start_time) #accumulo qui il tempo relativo all'ottimizzazione

    #aggiorno la matrice che conterrà gli alpha_star di ciascuna previsione binaria
    if alpha.size==0:
        alpha=alpha_star
    else:
        alpha=np.vstack((alpha,alpha_star))
    #print(alpha.shape) #3x1600


    K_train_red=rbf_kernel(X_train_red,X_train_red, gamma) 
    #aggiorno le liste per le stampe finali per stampe finali
    KKT_viol_list[i]=KKT_viol
    #Q_comp_rows_list.append(Q_computed_rows)
    optimal_f_list[i]=funz
    iteration_list[i]=iterations


    # creo la matrice con le previsioni convertendo le etichette binarie con i valori reali ( 1, 5, o 7 a seconda delle label in set che sto considerando)
    if predict_array.size==0:
        predict_array=predict(Y_train_red, K_train_red, rbf_kernel(X_train_red,X_test_all, gamma), alpha_star, C)
        predict_array=np.where(predict_array==1, set[1], predict_array)
        predict_array=np.where(predict_array==-1, set[0], predict_array)
    # aggiorno la matrice con le previsioni convertendo le etichette binarie con i valori reali (1,5, o 7 a seconda delle label in set che sto considerando))
    else:
        predict_array=np.vstack((predict_array,predict(Y_train_red, K_train_red, rbf_kernel(X_train_red,X_test_all, gamma), alpha_star, C))) 
        predict_array[i,:]=np.where(predict_array[i,:]==1, set[1], predict_array[i,:]) #converto le previsioni binarie nell'etichette vere associate
        predict_array[i,:]=np.where(predict_array[i,:]==-1, set[0], predict_array[i,:])


    # faccio lo stesso per calcolare l'accuratezza sul training
    # creo la matrice con le previsioni convertendo le etichette binarie con i valori reali ( 1, 5, o 7 a seconda delle label in set che sto considerando)
    if predict_array_train.size==0:
        predict_array_train=predict(Y_train_red, K_train_red, rbf_kernel(X_train_red,X_train_all, gamma), alpha_star, C)
        predict_array_train=np.where(predict_array_train==1, set[1], predict_array_train)
        predict_array_train=np.where(predict_array_train==-1, set[0], predict_array_train)
    # aggiorno la matrice con le previsioni convertendo le etichette binarie con i valori reali (1,5, o 7 a seconda delle label in set che sto considerando))
    else:
        predict_array_train=np.vstack((predict_array_train,predict(Y_train_red, K_train_red, rbf_kernel(X_train_red,X_train_all, gamma), alpha_star, C))) 
        predict_array_train[i,:]=np.where(predict_array_train[i,:]==1, set[1], predict_array_train[i,:]) #converto le previsioni binarie nell'etichette vere associate
        predict_array_train[i,:]=np.where(predict_array_train[i,:]==-1, set[0], predict_array_train[i,:])



#print(predict_array.shape) #3x600
#NB predic_array CONTIENE GIA' LE PREVISIONI CONVERTITE CON L'ETICHETTA VERA E NON è PIU' BINARIA
pred = np.apply_along_axis(voting, axis=0, arr=predict_array) #conta le occorrenze di ciascun label sulle colonne di predict_array e restituisce un array 1 x (numero sample del test) con il label con più occorrenze per ciascuna colonna 
## apply_along_axis lungo axis=0 applica la funzione voting su tutte le colonne dell'array predict_array
#calcolo l'error rate sul test set
error_rate = np.sum(pred != Y_test_all)/(Y_test_all.shape)

# faccio lo stesso per calcolare l'accuratezza sul training
pred_train = np.apply_along_axis(voting, axis=0, arr=predict_array_train)
error_rate_train = np.sum(pred_train != Y_train_all)/(Y_train_all.shape)



#CREAZIONE TABELLA OUTPUT
tabella=    [["1. C", C], ["2. Gamma", gamma], ["3. Implemented multiclass strategy", "OAO"], ["4. Accuracy on training set", 1-(error_rate_train[0])], 
["5. Accuracy on test set", 1-error_rate[0]], ["6. Run Time (seconds)", optimization_time], ["7. Number of iterations", sum(iteration_list)], 
["8. KKT violations (m-M) 1vs7",KKT_viol_list[0]], ["9. KKT violations (m-M) 5vs1",KKT_viol_list[1]], ["10. KKT violations (m-M) 7vs5",KKT_viol_list[2]], 
["11. Optimal f value 1vs7",(optimal_f_list[0])], ["12. Optimal f value 5vs1",(optimal_f_list[1])], ["13. Optimal f value 7vs5",(optimal_f_list[2])]]
                            
# Larghezza colonne e spaziatura manuale
col1_width = 40  # Larghezza fissa per la prima colonna
col2_width = 10  # Larghezza fissa per la seconda colonna
spaziatura = 5   # Spaziatura tra le colonne

# Stampa dell'intestazione
print(f"{''.ljust(col1_width)}{' ' * spaziatura}{'Q4 Values'.ljust(col2_width)}")

# Stampa delle righe della tabella
for riga in tabella:
    print(f"{str(riga[0]).ljust(col1_width)}{' ' * spaziatura}{str(riga[1]).ljust(col2_width)}")

print_confusion_matrix(pred, Y_test_all)

# ---------------------------------------OAO-----------------------------------------

'''
# ---------------------------------------OAA-----------------------------------------
arg_list=[]
arg_list_train=[]
optimal_f_list=[]
KKT_viol_list=[]
iteration_list=[]
optimization_time=0
for i, label in enumerate(true_label_list):
    start_time = time.time()
    X_train_OAA, Y_train_OAA = OAA(X_train_all, Y_train_all, label) #label contro tutte le altre etichette
    alpha_star, Q_computed_rows, iterations, KKT_viol, funz = decomposition_algorithm_MVP(X_train_OAA, Y_train_OAA, gamma, C) #risolvo il sottoproblema e trovo alpha_star
    end_time = time.time()
    optimization_time+= (end_time -start_time) #accumulo qui il tempo relativo all'ottimizzazione
    arg_list.append(predict_OAA(X_train_OAA, Y_train_OAA, rbf_kernel(X_train_OAA,X_test_all, gamma), alpha_star, gamma, C)) #appendo in una lista le previsioni del sottoproblema corrente
    arg_list_train.append(predict_OAA(X_train_OAA, Y_train_OAA, rbf_kernel(X_train_OAA,X_train_all, gamma), alpha_star, gamma, C)) #appendo in una lista le previsioni del sottoproblema corrente
    iteration_list.append(iterations)

    #aggiornamenti liste per stampe finali 
    KKT_viol_list.append(KKT_viol)
    #error_rate_train_list.append(error_rate_train)
    optimal_f_list.append(funz)

arg_array=np.array(arg_list) #lo trasfomo in un array 3 x numero di nuovi sample
indici_pred = np.argmax(arg_array, axis=0)
pred = np.array(true_label_list)[indici_pred]
error_rate = np.sum(pred != Y_test_all)/(Y_test_all.shape)

#calcoli per stampe training accuracy:
arg_array_train=np.array(arg_list_train) #lo trasfomo in un array 3 x numero di nuovi sample
indici_pred_train = np.argmax(arg_array_train, axis=0)
pred_train = np.array(true_label_list)[indici_pred_train]
error_rate_train = np.sum(pred_train != Y_train_all)/(Y_train_all.shape)

#CREAZIONE TABELLA OUTPUT
tabella=    [["1. C", C], ["2. Gamma", gamma], ["3. Implemented multiclass strategy", "OAA"], ["4. Accuracy on training set", 1-(error_rate_train[0])], 
["5. Accuracy on test set", 1-error_rate[0]], ["6. Run Time (seconds)", optimization_time], ["7. Number of iterations",(sum(iteration_list))], ["8. KKT violations (m-M) 1 vs all",KKT_viol_list[0]],
["9. KKT violations (m-M) 5 vs all",KKT_viol_list[1]], ["10. KKT violations (m-M) 7 vs all",KKT_viol_list[2]], ["11. Optimal f value 1 vs all",(optimal_f_list[0])],
["9. Optimal f value 5 vs all",(optimal_f_list[1])], ["10. Optimal f value 7 vs all",(optimal_f_list[2])]]
                            
# Larghezza colonne e spaziatura manuale
col1_width = 40  # Larghezza fissa per la prima colonna
col2_width = 10  # Larghezza fissa per la seconda colonna
spaziatura = 5   # Spaziatura tra le colonne

# Stampa dell'intestazione
print(f"{''.ljust(col1_width)}{' ' * spaziatura}{'Values'.ljust(col2_width)}")

# Stampa delle righe della tabella
for riga in tabella:
    print(f"{str(riga[0]).ljust(col1_width)}{' ' * spaziatura}{str(riga[1]).ljust(col2_width)}")

print_confusion_matrix(pred, Y_test_all)
# ---------------------------------------OAA-----------------------------------------
'''