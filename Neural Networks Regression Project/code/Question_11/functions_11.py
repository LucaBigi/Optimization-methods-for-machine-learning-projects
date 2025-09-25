import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

tol_minimize=10**-7
method_minimize="L-BFGS-B"


def load_data(file_path):
    np.random.seed(200)
    data = pd.read_csv(file_path) 
    x = data.iloc[:, :2].values  
    y = data.iloc[:, 2].values.reshape(-1, 1)  
    # Mescoliamo i dati - li mescoliamo per cercare di consentire al modello di generalizzare il più possibile - in realtà in questo caso non sarebbe necessario visto che i punti X nel dataset sono già generati in maniera random
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices) 
    x = x[indices]
    y = y[indices]
    return x, y


#FUNZIONI PER RETE MLP
def g(t, sigma):
    return np.tanh(t * sigma)

def predict(x_train, W, V, sigma):
    #x_train è un vettore contenente gli input x1,x2
    #W è una matrice con N righe e n+1 colonne contenente i pesi degli archi compresi tra input e hidden layer
    #V è un vettore riga con N colonne
    # Aggiungere la colonna di 1 agli input
    colonna_1 = np.ones((x_train.shape[0], 1))  # Colonna di 1 per gestire il bias
    X = np.hstack((x_train, colonna_1))  # Concateno una colonna di 1 agli input - risultato è una matrice con P righe e n+1 colonne
    z_1 = g(np.dot(W, X.T),sigma)  # --> Output dell'hidden layer, è una matrice con N righe e P colonne
    y_pred = np.dot(V, z_1)  # Output finale della MLP - è un vettore riga con P colonne
    return y_pred

def emp_error(x_train,true_y, W, V, sigma):
    colonna_1 = np.ones((x_train.shape[0], 1))
    X = np.hstack((x_train, colonna_1))
    z_1 = g(np.dot(W, X.T),sigma) 
    y_pred = np.dot(V, z_1)
    error = (1/2) * np.mean((y_pred - true_y.T) ** 2)    
    return error

def emp_error_reg(x_train, true_y, W, V, rho, sigma):
    reg_term = (rho / 2) * (np.linalg.norm(W) ** 2 + np.linalg.norm(V) ** 2)
    error = emp_error(x_train, true_y, W, V, sigma)
    return error + reg_term

def compute_gradient(x_train, true_y, W, V, rho, sigma):
    colonna_1 = np.ones((x_train.shape[0], 1))
    X = np.hstack((x_train, colonna_1)) 
    z_0 = X.T 
    z_1 = g(np.dot(W, z_0), sigma)  
    y_pred = np.dot(V, z_1) 
    P = y_pred.shape[1]
    delta = (1 / P) * (y_pred - true_y.T)
    grad_V = np.dot(delta, z_1.T) + rho * V
    derivata_g = sigma * (1 - g(np.dot(W, z_0), sigma) ** 2)
    grad_W = np.dot((delta.T * V).T * derivata_g, z_0.T) + rho * W
    grad = np.hstack((grad_W.flatten(), grad_V.flatten()))
    return grad

# Funzione per effettuare l'ottimizzazione dei pesi
def train_mlp(x_train, true_y, N, rho, sigma,seme):
    n_inputs = x_train.shape[1] 
    n_outputs = true_y.shape[1]
    np.random.seed(seme)
    W_init = np.random.randn(N, n_inputs + 1)
    V_init = np.random.randn(n_outputs, N)
    initial_params = np.hstack((W_init.flatten(), V_init.flatten()))
    
    def objective(params): #creo una nuova funzione che prende in ingresso il vettore params e ricostruisce le matrici W e di V per calcolare l'errore regolarizzato
        W = params[:N * (n_inputs + 1)].reshape(N, n_inputs + 1) 
        V = params[N * (n_inputs + 1):].reshape(n_outputs, N)
        return emp_error_reg(x_train, true_y, W, V, rho, sigma)
    
    def objective_grad(params): #creo una nuova funzione che prende in ingresso il vettore params e ricostruisce le matrici W e di V per calcolare il gradiente dell'errore regolarizzato
        W = params[:N * (n_inputs + 1)].reshape(N, n_inputs + 1)
        V = params[N * (n_inputs + 1):].reshape(n_outputs, N)
        return compute_gradient(x_train, true_y, W, V, rho, sigma)   
       
    result = minimize(objective, initial_params, method=method_minimize, jac=objective_grad, tol=tol_minimize)  
           
    # Estrai i pesi ottimizzati e ricostruisci la matrice W e il vettore V
    optimized_params = result.x
    W_opt = optimized_params[:N * (n_inputs + 1)].reshape(N, n_inputs + 1)
    V_opt = optimized_params[N * (n_inputs + 1):].reshape(n_outputs, N)
    initial_objective_value = objective(initial_params)
    initial_gradient = objective_grad(initial_params)
    return W_opt, V_opt, result.fun, result.message, result.nit, result.nfev, result.jac, initial_objective_value, initial_gradient, result.njev



#### FUNZIONE PER K-FOLD
#def k_fold_cross_validation(x, y, k, N, rho, sigma,seme):
#    n_samples = x.shape[0] 
#    fold_size = n_samples // k # Calcolo la grandezza del fold
#    indices = np.arange(n_samples) 
#    train_errors = []  # Lista per memorizzare gli errori sul training set (senza regolarization term)
#    validation_errors = []  # Lista per memorizzare gli errori per ogni fold (senza regolarization term)
#    train_errors_reg = []  # Lista per memorizzare gli errori sul training set (con regolarization term)
#    validation_errors_reg = []  # Lista per memorizzare gli errori per ogni fold (con regolarization term)

#    for i in range(k):
#        validation_indices = indices[i * fold_size:(i + 1) * fold_size] 
#        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
#        x_train = x[train_indices]
#        y_train = y[train_indices]
#        x_validation = x[validation_indices]
#        y_validation = y[validation_indices]
#        # Addestro il modello sul trainig set e trovo la configurazione ottima di pesi
#        W_opt, V_opt,_,_,_,_,_,_,_,_  = train_mlp(x_train, y_train, N, rho, sigma,seme)       
#        validation_error = emp_error(x_validation, y_validation, W_opt, V_opt, sigma)
#        validation_errors.append(validation_error)
#        validation_error_reg = emp_error_reg(x_validation, y_validation, W_opt, V_opt, rho, sigma)
#        validation_errors_reg.append(validation_error_reg)
#        train_error = emp_error(x_train, y_train, W_opt, V_opt, sigma)
#        train_errors.append(train_error)
#        train_error_reg = emp_error_reg(x_train, y_train, W_opt, V_opt, rho, sigma)
#        train_errors_reg.append(train_error_reg)

#    mean_validation_error = np.mean(validation_errors)
#    mean_validation_error_reg = np.mean(validation_errors_reg) 
#    mean_training_error = np.mean(train_errors)
#    mean_training_error_reg = np.mean(train_errors_reg)
#    return mean_validation_error, mean_training_error, mean_validation_error_reg, mean_training_error_reg


####FUNZIONI PER GRAFICARE L'ANDAMENTO DELL'ERRORE (TRAINING E VALIDATION) IN FUNZIONE DEGLI IPERPARAMETRI
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
#    plt.title(f'Graph 1\n(seed={optimal_seed}, rho={optimal_rho}, sigma={optimal_sigma})')
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
#    plt.title(f'Graph 2\n(seed={optimal_seed}, N={optimal_N}, sigma={optimal_sigma})')
#    plt.legend()
#    plt.grid()
#    plt.show()

    # Grafico 3:
#    df_D = df_A[(df_A['Numero di neuroni N'] == optimal_N) & (df_A['rho'] == optimal_rho)]
#    plt.figure()
#    plt.plot(df_D['sigma'], df_D['mean_validation_error'], marker='o', label='Mean Validation Error')
#    plt.plot(df_D['sigma'], df_D['mean_training_error'], marker='x', label='Mean Training Error', color='orange')
#    plt.xlabel('sigma')
#    plt.ylabel('Error')
#    plt.title(f'Graph 3\n(seed={optimal_seed}, N={optimal_N}, rho={optimal_rho})')
#    plt.legend()
#    plt.grid()
#    plt.show()


#def GraficiIperparametri(file_path_N, file_path_rho, file_path_sigma, seed, N, rho, sigma):  
#    #insierisci nella chiamata funzione i 3 path file seguiti da gli iperparametri N, rho, sigma scelti
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


def grafico(W_opt, V_opt, sigma,filename):    
    x1_grid = np.linspace(-2, 2, 100)
    x2_grid = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    y_pred_grid = predict(grid_points, W_opt, V_opt, sigma)
    Y_pred = y_pred_grid.reshape(X1.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y_pred, cmap='viridis', edgecolor='none')
    ax.set_title('Neural Network MLP output')
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



def test(test_dataset, W_opt, V_opt,sigma):
    x_test,y_test=load_data(test_dataset)
    # Calcola l'errore sul test set 
    test_error = emp_error(x_test, y_test, W_opt, V_opt, sigma)
    return test_error