from functions_12 import *

## STEP 1 GRID SEARCH
##NB: I valori degli iperparametri scelti per l'ottimizzazione finale non sono quelli selezionati dalla grid_search, bensì sono dei valori scelti successivamente attraverso l'analisi dei grafici dell'ANDAMENTO DELLL'ERRORE IN FUNZIONE DEGLI IPERPARAMETRI che prendono come input i risultati della grid search. Da questi grafici è visibile che la complessità del modello può essere diminuita rispetto a quella scelta dalla grid_search a vantaggio di un ottimizzazione più rapida
##NB: Dopo una prima analisi con una primissima grid search abbiamo deciso di scegliere rho1=rho2=rho poichè i valori ottimi per rho tendevano ad essere in entrambi i casi su 10**-5
#valori_rho=[10**(-4.5),10**(-4.75), 10**(-5)] 
#valori_sigma=[0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95] 
#k=5 #numero di fold per k-cross validation
#minimo_validation=[99,99,99,99, None, None, None]
#err_list=[]
#x,y=load_data("dataset.csv") 

#for seme in range (100,301,100): #<-- seme ottimo - ciclo per grid-search --> #(100,301,100):  ciclo per ricerca seme ottimo --> #(100,10000,76):
#    for N in range (40, 70, 1):
#        for rho in valori_rho:   
#            for sigma in valori_sigma:                              
#                mean_validation_error, mean_training_error, mean_validation_error_reg, mean_training_error_reg = k_fold_cross_validation_rbf(x, y, k, N, rho,rho, sigma,seme)
#                if minimo_validation[0]>mean_validation_error:
#                    minimo_validation=[mean_validation_error,seme,N,rho,sigma]                 
#                err_list.append([mean_validation_error, mean_training_error, mean_validation_error_reg, mean_training_error_reg,seme,N,rho,sigma])                     
#print ("combinazione imperparametri con minimo validation_error senza regolarizzazione -> (mean_validation_error, seme, N, rho, sigma) =", minimo_validation)
#df1 = pd.DataFrame(err_list, columns=['mean_validation_error', 'mean_training_error', 'mean_validation_error_reg', 'mean_training_error_reg','seme', 'Numero di neuroni N', 'rho', 'sigma'])
#df1.to_csv("TabellaErrori_rbf_GridSearch.csv", index=False)
#Grafici_Grid_Search('TabellaErrori_rbf_GridSearch.csv')

## Analizzo Grafici_Grid_Search --> Scelta iperparametri --> rho1=rho2=rho=10*-5 , sigma=0.88, N=50
## Ciclo sul seme per trovare un buon seme --> (100,10000,76) --> seme con minimo mean validation error --> 2532

## STEP 2: fissamo i valori di due iperparametri ai valori scelti allo step precedente e poi visualizziamo l'andamento dell'errore (training e validation) in funzione dell'iperparametro che rimaneva variabile sul ciclo for 

##Ricavo il primo csv:
#valori_rho=[10**(-3),10**(-3.25),10**(-3.5),10**(-3.75), 10**(-4),10**(-4.25),10**(-4.5),10**(-4.75), 10**(-5)]
#valori_sigma= [0.88]
#   .....
#for seme in range (2532,2533,1):  
#    for N in range (50, 51, 1):  
#        for rho in valori_rho:   
#            for sigma in valori_sigma:  
#   .....
#df1 = pd.DataFrame(err_list, columns=['mean_validation_error', 'mean_training_error', 'mean_validation_error_reg', 'mean_training_error_reg','seme', 'Numero di neuroni N', 'rho', 'sigma'])
#df1.to_csv("TabellaErrori_rbf_rho.csv", index=False)

##Ricavo il secondo csv:
#valori_rho=[10**(-5)]
#valori_sigma= [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]
#   .....
#for seme in range (2532,2533,1):  
#    for N in range (50, 51, 1):  
#        for rho in valori_rho:   
#            for sigma in valori_sigma:  
#   .....
#df1 = pd.DataFrame(err_list, columns=['mean_validation_error', 'mean_training_error', 'mean_validation_error_reg', 'mean_training_error_reg','seme', 'Numero di neuroni N', 'rho', 'sigma'])
#df1.to_csv("TabellaErrori_rbf_sigma.csv", index=False)

##Ricavo il terzo csv:
#valori_rho=[10**(-5)]
#valori_sigma= [0.88]
#   .....
#for seme in range (2532,2533,1):  
#    for N in range (20, 100, 10):  
#        for rho in valori_rho:   
#            for sigma in valori_sigma:  
#   .....
#df1 = pd.DataFrame(err_list, columns=['mean_validation_error', 'mean_training_error', 'mean_validation_error_reg', 'mean_training_error_reg','seme', 'Numero di neuroni N', 'rho', 'sigma'])
#df1.to_csv("TabellaErrori_rbf_N.csv", index=False)

seme=2532
N=50
rho=10**(-5)
sigma=0.88
#GraficiIperparametri("TabellaErrori_rbf_N.csv", "TabellaErrori_rbf_rho.csv", "TabellaErrori_rbf_sigma.csv", seme, N, rho, sigma)
## Scelta iperparametri finale --> rho1=rho2=rho=10*-5 , sigma=0.88, N=50


#TRAINING
x,y=load_data("dataset.csv")
# Addestro il modello sul trainig set e trovo la configurazione ottima di pesi
start_time = time.time()
centers_opt, v_opt,optimal_f,output_message,niter, n_fun_eval, final_gradient, initial_objective_value, initial_gradient, n_grad_eval   = train_rbf(x, y, N, rho, rho, sigma,seme)
end_time = time.time()
optimization_time = end_time - start_time
# Calcola l'errore sul training set 
train_error = emp_error_rbf(x, y, centers_opt, v_opt, sigma)  
x_validation, y_validation = validation_set(x, y, 20)  
validation_error = emp_error_rbf(x_validation, y_validation, centers_opt, v_opt, sigma) 

#CREAZIONE TABELLA OUTPUT
tabella=    [["1. Neurons", N], ["2. Rho", rho],["3. Sigma", sigma], ["4. Gradient Tollerance", tol_minimize], ["5. Optimization Solver", method_minimize],
             ["6. Output Message",output_message],["7. Starting f value", initial_objective_value], ["8. Optimal f value",optimal_f], ["9. Gradient Norm in starting point", np.linalg.norm(initial_gradient)],
             ["10. Gradient Norm in optimal point",np.linalg.norm(final_gradient)], ["11. Iterations", niter], ["12. Function evaluations",n_fun_eval],
             ["13. Gradient evaluations", n_grad_eval], ["14. Optimal Time (seconds)",optimization_time], ["15. Training Error", train_error],[ "16. Validation Error", validation_error]]
                            
# Larghezza colonne e spaziatura manuale
col1_width = 40  # Larghezza fissa per la prima colonna
col2_width = 10  # Larghezza fissa per la seconda colonna
spaziatura = 5   # Spaziatura tra le colonne

# Stampa dell'intestazione
print(f"{''.ljust(col1_width)}{' ' * spaziatura}{'Values'.ljust(col2_width)}")

# Stampa delle righe della tabella
for riga in tabella:
    print(f"{str(riga[0]).ljust(col1_width)}{' ' * spaziatura}{str(riga[1]).ljust(col2_width)}")

grafico(centers_opt, v_opt, sigma,"img_12.png")


#TEST
test_dataset="blind_test.csv"
print("L'errore sul test se è:", test(test_dataset, centers_opt, v_opt,sigma))