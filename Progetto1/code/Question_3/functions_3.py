import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import time
import os

tol_minimize=10**-5

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
    N=centers.shape[0]
    v=v.reshape(N,1)
    y_pred= np.dot(phi(x, centers, sigma), v)
    return y_pred


def emp_error_rbf(x, true_y, centers, v, sigma):
    P = x.shape[0]
    y_pred=predict_rbf(x, centers, v, sigma) # dimensione Px1
    return (1 / (2 * P)) * np.sum((y_pred - true_y) ** 2)

def emp_error_reg_rbf(x, true_y, centers, v, rho, sigma):
    error=emp_error_rbf(x, true_y, centers, v, sigma)
    reg_term = (rho/2) * np.linalg.norm(v) ** 2 + (rho/2) * np.linalg.norm(centers.flatten()) ** 2 
    return error + reg_term

def hessE(x, centers,N,P,sigma,rho):     
     phi_val=phi(x, centers, sigma)     
     Q = (1/P) * np.dot(phi_val.T,phi_val) + rho*np.eye(N)    
     return Q

def coeff(x, y, centers,N,P,sigma): 
    phi_val=phi(x, centers, sigma)
    c = (1/P) * np.dot(phi_val.T,y)
    return c

def train_rbf_pesi(x, y, centers, N, rho, sigma):
    P=x.shape[0]
    Q = hessE(x, centers,N,P,sigma,rho)
    c = coeff(x, y, centers,N,P,sigma)
    v_star, residuals, rank, eigenvectors = np.linalg.lstsq(Q,c,rcond=None)
    return v_star.T


def compute_gradient_c_rbf(x, true_y, centers, v, rho, sigma):
    P = x.shape[0]    
    c1, c2 = centers[:,0] , centers[:,1]    
    errore=predict_rbf(x, centers, v, sigma) - true_y #Px1
    phi_val = phi(x, centers, sigma) #PxN
    x1_diff = x[:, 0][:, np.newaxis] - c1 #matrice PxN  
    x2_diff = x[:, 1][:, np.newaxis] - c2 #matrice PxN
    somma_p1 = np.multiply((np.multiply(errore,phi_val)),x1_diff).sum(axis=0) #Nx1
    somma_p2 = np.multiply((np.multiply(errore,phi_val)),x2_diff).sum(axis=0) #Nx1
    grad_c1 = (2 / (P * (sigma ** 2))) * somma_p1 * v + rho * c1
    grad_c2 = (2 / (P * (sigma ** 2))) * somma_p2 * v + rho * c2
    grad_c = np.vstack((grad_c1,grad_c2)).T  #Nxn
     
    return grad_c.flatten()

def compute_gradient_rbf(x, true_y, centers, v, rho, sigma):
    P = x.shape[0]    
    c1, c2 = centers[:,0] , centers[:,1]    
    errore=predict_rbf(x, centers, v, sigma) - true_y #Px1
    phi_val = phi(x, centers, sigma) #PxN
    grad_v = (1 / P) * np.dot(errore.T, phi_val) + rho * v #1xN
    grad_c = compute_gradient_c_rbf(x, true_y, centers, v, rho, sigma)   
 
    return np.concatenate([grad_c, grad_v.flatten()])

def train_rbf(x_train, true_y, centers, N, v, rho, sigma):
    n = x_train.shape[1]    
    initial_params = centers.flatten()
    
    def objective(params):
        centers = params[:N*n].reshape(N, n)
        return emp_error_reg_rbf(x_train, true_y, centers, v, rho, sigma)
    
    def objective_grad(params):
        centers = params[:N*n].reshape(N, n)
        return compute_gradient_c_rbf(x_train, true_y, centers, v, rho, sigma)

    result = minimize(objective, initial_params, method='L-BFGS-B', jac=objective_grad, tol=tol_minimize)

    optimized_params = result.x
    centers_opt = optimized_params[:N*2].reshape(N, n)
    return centers_opt, result.fun, result.success


def grafico (centers,v_opt,sigma,filename):
    x1_grid = np.linspace(-2, 2, 100)
    x2_grid = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    y_pred_grid = predict_rbf(grid_points, centers, v_opt, sigma)  
    Y_pred = y_pred_grid.reshape(X1.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y_pred, cmap='viridis', edgecolor='none')
    ax.set_title('Neural Network RBF output')
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






