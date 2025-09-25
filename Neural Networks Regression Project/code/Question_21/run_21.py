from functions_21 import *

N=100
sigma=2
rho=10**-5
x,y = load_data("dataset.csv")
P=x.shape[0]

##MULTISTART:
#seme_iniziale=1
#seme_minimo = seme_iniziale
#min_train_error_reg=99
#seme=seme_iniziale
#for i in range (1,100000):
#    V_opt, W  = train_mlp_unsupervised(x, y, N, rho, sigma,seme)
#    train_error_reg = emp_error_reg(x, y, W, V_opt, rho, sigma)
#    if train_error_reg < min_train_error_reg:
#        min_train_error_reg=train_error_reg
#        seme_minimo=seme
#    seme = seme_iniziale + i
#print(seme_minimo)

seme_minimo=33410  
start_time = time.time()  
V_opt, W = train_mlp_unsupervised(x, y, N, rho, sigma,seme_minimo)
end_time = time.time()
optimization_time = end_time - start_time

optimal_f=emp_error_reg(x, y, W, V_opt, rho, sigma)
final_gradient=compute_gradient(x, y, W, V_opt, rho, sigma)
error_seme_minimo = emp_error(x, y, W, V_opt, sigma)
x_validation, y_validation = validation_set(x, y, 20)
validation_error = emp_error(x_validation, y_validation, W, V_opt, sigma)  

#CREAZIONE TABELLA OUTPUT
tabella=    [["1. Neurons", N], ["2. Rho", rho],["3. Sigma", sigma], ["4. Gradient Tollerance", "not available"], ["5. Optimization Solver", method_minimize],
             ["6. Output Message","not available"],["7. Starting f value", "not available"], ["8. Optimal f value",optimal_f], ["9. Gradient Norm in starting point", "not available"],
             ["10. Gradient Norm in optimal point",np.linalg.norm(final_gradient)], ["11. Iterations", "not available"], ["12. Function evaluations","not available"],
             ["13. Gradient evaluations", "not available"], ["14. Optimal Time (seconds)",optimization_time], ["15. Training Error", error_seme_minimo],[ "16. Validation Error", validation_error]]
                            
# Larghezza colonne e spaziatura manuale
col1_width = 40  # Larghezza fissa per la prima colonna
col2_width = 10  # Larghezza fissa per la seconda colonna
spaziatura = 5   # Spaziatura tra le colonne

# Stampa dell'intestazione
print(f"{''.ljust(col1_width)}{' ' * spaziatura}{'Values'.ljust(col2_width)}")

# Stampa delle righe della tabella
for riga in tabella:
    print(f"{str(riga[0]).ljust(col1_width)}{' ' * spaziatura}{str(riga[1]).ljust(col2_width)}")


grafico(W, V_opt, sigma,"img_21.png")

#TEST ERROR
test_dataset="blind_test.csv"
print("L'errore sul test se è:", test(test_dataset, W, V_opt,sigma))





