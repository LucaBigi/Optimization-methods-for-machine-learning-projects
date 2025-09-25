from functions_3 import *

seme=94608 #seme trovato nell'esercizio 2.2
N=50
sigma=0.88
rho=10**-5
x,y = load_data("dataset.csv")
P=x.shape[0]
max_iter=500
patience_max=10
grad_tol = 10**-5
k=0
patience=0

np.random.seed(seme)
indici_casuali = np.random.choice(x.shape[0], N, replace=False)
centers_init = x[indici_casuali]
v_init = np.random.randn(N)   
gradient_init=compute_gradient_rbf(x, y, centers_init, v_init, rho, sigma)
norm_grad_init=np.linalg.norm(gradient_init)
f_init=emp_error_reg_rbf(x, y, centers_init, v_init, rho, sigma)
norm_grad=norm_grad_init

start_time=time.time()
while k < max_iter and patience<patience_max and norm_grad > grad_tol:
    if k==0:
        centers=centers_init
        v_opt=v_init
    train_error_reg0 = emp_error_reg_rbf(x, y, centers, v_opt, rho, sigma)
    v_opt = train_rbf_pesi(x, y, centers, N, rho, sigma)
    centers, _ , _ = train_rbf(x, y, centers, N, v_opt, rho, sigma)
    train_error_reg1 = emp_error_reg_rbf(x, y, centers, v_opt, rho, sigma)
    gradient=compute_gradient_c_rbf(x, y, centers, v_opt, rho, sigma)
    norm_grad=np.linalg.norm(gradient)
    if train_error_reg0 - train_error_reg1 < 10**-8:
        patience = patience + 1
    k=k+1
end_time = time.time()
optimization_time = end_time - start_time

optimal_f=emp_error_reg_rbf(x, y, centers, v_opt, rho, sigma)
final_gradient=compute_gradient_rbf(x, y, centers, v_opt, rho, sigma)
training_error= emp_error_rbf(x, y, centers, v_opt, sigma)
x_validation, y_validation = validation_set(x, y, 20)
validation_error = emp_error_rbf(x_validation, y_validation, centers, v_opt, sigma)  

#CREAZIONE TABELLA OUTPUT
tabella=    [["1. Neurons", N], ["2. Rho", rho],["3. Sigma", sigma], ["4. Gradient Tollerance", tol_minimize], ["5. Optimization Solver", "L-BFGS-B and np.linalg.lstsq"],
             ["6. Output Message","not available"],["7. Starting f value", f_init], ["8. Optimal f value",optimal_f], ["9. Gradient Norm in starting point", norm_grad_init],
             ["10. Gradient Norm in optimal point",norm_grad], ["11. Iterations", k], ["12. Function evaluations","not available"],
             ["13. Gradient evaluations", "not available"], ["14. Optimal Time (seconds)",optimization_time], ["15. Training Error", training_error],[ "16. Validation Error", validation_error]]
                            
# Larghezza colonne e spaziatura manuale
col1_width = 40  # Larghezza fissa per la prima colonna
col2_width = 10  # Larghezza fissa per la seconda colonna
spaziatura = 5   # Spaziatura tra le colonne

# Stampa dell'intestazione
print(f"{''.ljust(col1_width)}{' ' * spaziatura}{'Values'.ljust(col2_width)}")

# Stampa delle righe della tabella
for riga in tabella:
    print(f"{str(riga[0]).ljust(col1_width)}{' ' * spaziatura}{str(riga[1]).ljust(col2_width)}")


grafico (centers,v_opt,sigma,"img_3.png")

#TEST ERROR
test_dataset="blind_test.csv"
print("L'errore sul test se è:", test(test_dataset, centers, v_opt,sigma))


