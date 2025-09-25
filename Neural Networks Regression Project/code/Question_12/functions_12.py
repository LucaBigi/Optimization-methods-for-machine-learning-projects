import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

import os
import time

tol_minimize=10**-7
method_minimize="L-BFGS-B"

def load_data(file_path):
    np.random.seed(200) 
    data = pd.read_csv(file_path)  
    x = data.iloc[:, :2].values  
    y = data.iloc[:, 2].values.reshape(-1, 1) 
    indices = np.arange(x.shape[0]) 
    np.random.shuffle(indices)  
    x = x[indices] 
    y = y[indices]
    return x, y

def phi(x, centers, sigma):
    x1, x2 = x[:, 0][:, np.newaxis] , x[:, 1][:, np.newaxis] #due vettori Px1
    c1, c2 = centers[:,0] , centers[:,1] #due vettori 1xN
    norm_dist_sqr = (x1 - c1)**2 +(x2 - c2)**2  # x1-c1 --> Px1 - 1xN --> broadcast PxN (duplico i valori su dimensione N) - PxN (duplico i valori su dimensione P) --> il risultato è PxN con ogni punto (prima cordinata) sottratto ad ogni (prima cordinata)
    return  np.exp(-norm_dist_sqr / (sigma ** 2)) #matrice PxN

def predict_rbf(x, centers, v, sigma):
    P = x.shape[0]
    N=centers.shape[0]
    v=v.reshape(N,1)
    y_pred= np.dot(phi(x, centers, sigma), v)
    return y_pred

def emp_error_rbf(x, true_y, centers, v, sigma):
    P = x.shape[0]
    y_pred=predict_rbf(x, centers, v, sigma) # dimensione Px1
    return (1 / (2 * P)) * np.sum((y_pred - true_y) ** 2)

def emp_error_reg_rbf(x, true_y, centers, v, rho1,rho2, sigma):
    error=emp_error_rbf(x, true_y, centers, v, sigma)
    reg_term = (rho1/2) * np.linalg.norm(v) ** 2 + (rho2/2) * np.linalg.norm(centers.flatten()) ** 2 
    return error + reg_term

def compute_gradient_rbf(x, true_y, centers, v, rho1,rho2, sigma):
    P = x.shape[0]    
    c1, c2 = centers[:,0] , centers[:,1]    
    errore=predict_rbf(x, centers, v, sigma) - true_y #Px1
    phi_val = phi(x, centers, sigma) #PxN
    grad_v = (1 / P) * np.dot(errore.T, phi_val) + rho1 * v #1xN
    x1_diff = x[:, 0][:, np.newaxis] - c1 #matrice PxN  
    x2_diff = x[:, 1][:, np.newaxis] - c2 #matrice PxN
    somma_p1 = np.multiply((np.multiply(errore,phi_val)),x1_diff).sum(axis=0) #Nx1
    somma_p2 = np.multiply((np.multiply(errore,phi_val)),x2_diff).sum(axis=0) #Nx1
    grad_c1 = (2 / (P * (sigma ** 2))) * somma_p1 * v + rho2 * c1
    grad_c2 = (2 / (P * (sigma ** 2))) * somma_p2 * v + rho2 * c2
    grad_c = np.vstack((grad_c1,grad_c2)).T  #Nxn
    return np.concatenate([grad_c.flatten(), grad_v.flatten()])



def train_rbf(x_train, true_y, N, rho1,rho2, sigma,seme):
    n = x_train.shape[1]    
    np.random.seed(seme) 
    centers = np.random.randn(N, n) 
    v = np.random.randn(N)    
    initial_params = np.hstack([centers.flatten(), v.flatten(),])
    
    def objective(params):
        centers = params[:N*n].reshape(N, n)
        v = params[N*n:]
        return emp_error_reg_rbf(x_train, true_y, centers, v, rho1,rho2, sigma)
    
    def objective_grad(params):
        centers = params[:N*n].reshape(N, n)
        v = params[N*n:]
        return compute_gradient_rbf(x_train, true_y, centers, v, rho1,rho2, sigma)


    result = minimize(objective, initial_params, method=method_minimize, jac=objective_grad, tol=tol_minimize)

    optimized_params = result.x
    centers_opt = optimized_params[:N*2].reshape(N, n)
    v_opt = optimized_params[N*2:]
    initial_objective_value = objective(initial_params)
    initial_gradient = objective_grad(initial_params)
    return centers_opt, v_opt, result.fun, result.message, result.nit, result.nfev, result.jac, initial_objective_value, initial_gradient, result.njev


### FUNZIONE PER K-FOLD
#def k_fold_cross_validation_rbf(x, y, k, N, rho1,rho2, sigma,seme):
#    n_samples = x.shape[0] 
#    fold_size = n_samples // k
#    indices = np.arange(n_samples)
#    train_errors = []  
#    validation_errors = [] 
#    train_errors_reg = [] 
#    validation_errors_reg = [] 

#    for i in range(k):
#        validation_indices = indices[i * fold_size:(i + 1) * fold_size] 
#        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
#        x_train = x[train_indices]
#        y_train = y[train_indices]
#        x_validation = x[validation_indices]
#        y_validation = y[validation_indices]
#        # Addestro il modello sul trainig set e trovo la configurazione ottima di pesi
#        centers_opt, v_opt,_,_,_,_,_,_,_,_  = train_rbf(x_train, y_train, N, rho1,rho2, sigma,seme)   
#        validation_error = emp_error_rbf(x_validation, y_validation, centers_opt, v_opt, sigma)
#        validation_errors.append(validation_error)
#        validation_error_reg = emp_error_reg_rbf(x_validation, y_validation, centers_opt, v_opt, rho1,rho2, sigma)
#        validation_errors_reg.append(validation_error_reg)
#        train_error = emp_error_rbf(x_train, y_train, centers_opt, v_opt, sigma)
#        train_errors.append(train_error)
#        train_error_reg = emp_error_reg_rbf(x_train, y_train, centers_opt, v_opt, rho1,rho2, sigma)
#        train_errors_reg.append(train_error_reg)
#    mean_validation_error = np.mean(validation_errors)
#    mean_validation_error_reg = np.mean(validation_errors_reg)
#    mean_training_error = np.mean(train_errors)
#    mean_training_error_reg = np.mean(train_errors_reg)
#    return mean_validation_error, mean_training_error, mean_validation_error_reg, mean_training_error_reg


###FUNZIONI PER GRAFICARE L'ANDAMENTO DELL'ERRORE (TRAINING E VALIDATION) IN FUNZIONE DEGLI IPERPARAMETRI
#def Grafici_Grid_Search(file_path):
#    df = pd.read_csv(file_path)
#    # Trova la riga con il minimo valore di mean_validation_error
#    min_row = df.loc[df['mean_validation_error'].idxmin()]
#    # Estrai i parametri ottimi
#    optimal_seed = min_row['seme']
#    optimal_N = min_row['Numero di neuroni N']
#    optimal_rho = min_row['rho']
#    optimal_sigma = min_row['sigma']
#    # Filtra la tabella per mantenere solo le righe con il seme ottimo (Tabella A)
#    df_A = df[df['seme'] == optimal_seed]
    
#    # Grafico 1:
#    df_B = df_A[(df_A['rho'] == optimal_rho) & (df_A['sigma'] == optimal_sigma)]
#    plt.figure()
#    plt.plot(df_B['Numero di neuroni N'], df_B['mean_validation_error'], marker='o', label='Mean Validation Error')
#    plt.plot(df_B['Numero di neuroni N'], df_B['mean_training_error'], marker='x', label='Mean Training Error', color='orange')
#    plt.xlabel('Numero di neuroni N')
#    plt.ylabel('Error')
#    plt.title(f'Graph 1 \n(seed={optimal_seed}, rho={optimal_rho}, sigma={optimal_sigma})')
#    plt.legend()
#    plt.grid()
#    plt.show()

#    # Grafico 2:
#    df_C = df_A[(df_A['Numero di neuroni N'] == optimal_N) & (df_A['sigma'] == optimal_sigma)]
#    plt.figure()
#    plt.plot(df_C['rho'], df_C['mean_validation_error'], marker='o', label='Mean Validation Error')
#    plt.plot(df_C['rho'], df_C['mean_training_error'], marker='x', label='Mean Training Error', color='orange')
#    plt.xlabel('rho')
#    plt.ylabel('Error')
#    plt.title(f'Graph 2 \n(seed={optimal_seed}, N={optimal_N}, sigma={optimal_sigma})')
#    plt.legend()
#    plt.grid()
#    plt.show()

#    # Grafico 3:
#    df_D = df_A[(df_A['Numero di neuroni N'] == optimal_N) & (df_A['rho'] == optimal_rho)]
#    plt.figure()
#    plt.plot(df_D['sigma'], df_D['mean_validation_error'], marker='o', label='Mean Validation Error')
#    plt.plot(df_D['sigma'], df_D['mean_training_error'], marker='x', label='Mean Training Error', color='orange')
#    plt.xlabel('sigma')
#    plt.ylabel('Error')
#    plt.title(f'Graph 3 \n(seed={optimal_seed}, N={optimal_N}, rho={optimal_rho})')
#    plt.legend()
#    plt.grid()
#    plt.show()



#def GraficiIperparametri(file_path_N, file_path_rho, file_path_sigma, seed, N, rho, sigma):
#    # Carica i file CSV 
#    #primo file: N variabile e sigma e rho fissati ai valori scelti
#    df1 = pd.read_csv(file_path_N)
#    #secondo file: rho variabile e N e rho fissati ai valori scelti
#    df2 = pd.read_csv(file_path_rho)
#    #terzo file: N variabile e sigma e rho fissati ai valori scelti
#    df3 = pd.read_csv(file_path_sigma)

#    # Grafico 1: 
#    plt.figure()
#    plt.plot(df1['Numero di neuroni N'], df1['mean_validation_error'], marker='o', label='Mean Validation Error')
#    plt.plot(df1['Numero di neuroni N'], df1['mean_training_error'], marker='x', label='Mean Training Error', color='orange')
#    plt.xlabel('N')
#    plt.ylabel('Error')
#    plt.title(f'Graph 1\n(seed={seed}, rho={rho}, sigma={sigma})')
#    plt.legend()
#    plt.grid()
#    plt.show()

#    # Grafico 2: 
#    plt.figure()
#    plt.plot(df2['rho'], df2['mean_validation_error'], marker='o', label='Mean Validation Error')
#    plt.plot(df2['rho'], df2['mean_training_error'], marker='x', label='Mean Training Error', color='orange')
#    plt.xlabel('rho')
#    plt.ylabel('Error')
#    plt.title(f'Graph 2\n(seed={seed}, N={N}, sigma={sigma})')
#    plt.legend()
#    plt.grid()
#    plt.show()

#    # Grafico 3: 
#    plt.figure()
#    plt.plot(df3['sigma'], df3['mean_validation_error'], marker='o', label='Mean Validation Error')
#    plt.plot(df3['sigma'], df3['mean_training_error'], marker='x', label='Mean Training Error', color='orange')
#    plt.xlabel('sigma')
#    plt.ylabel('Error')
#    plt.title(f'Graph 3\n(seed={seed}, N={N}, rho={rho})')
#    plt.legend()
#    plt.grid()
#    plt.show()

def grafico(centers_opt, v_opt, sigma,filename):
    x1_grid = np.linspace(-2, 2, 100)
    x2_grid = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    y_pred_grid = predict_rbf(grid_points, centers_opt, v_opt, sigma)
    Y_pred = y_pred_grid.reshape(X1.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y_pred, cmap='viridis', edgecolor='none')
    ax.set_title('Neural Network RBF Output')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Predicted output')
    plt.savefig(os.path.join(os.getcwd(), filename), format='png')
    plt.close(fig)

def validation_set(x, y, alpha): #alpha è la percentuale di dati per validation set 
    np.random.seed(800)
    n_samples = x.shape[0]
    test_size = (n_samples // 100) * alpha # Calcolo la grandezza del test set
    indices = np.arange(n_samples) 
    np.random.shuffle(indices)  # Mescolo randomicamente gli indici
    validation_indices = indices[n_samples-test_size:]
    x_validation = x[validation_indices]
    y_validation = y[validation_indices]
    return x_validation,y_validation

def test(test_dataset,centers_opt,v_opt,sigma):
    x_test,y_test=load_data(test_dataset)
    test_error = emp_error_rbf(x_test, y_test, centers_opt, v_opt, sigma)
    return test_error