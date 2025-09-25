import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

method_minimize="np.linalg.lstsq"

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


def g(t, sigma):
    return np.tanh(t * sigma)


def predict(x_train, W, V, sigma):
    colonna_1 = np.ones((x_train.shape[0], 1))
    X = np.hstack((x_train, colonna_1))
    z_1 = g(np.dot(W, X.T),sigma)
    y_pred = np.dot(V, z_1)
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

def hessE(x, W, rho, sigma, N):
    P=x.shape[0]
    colonna_1 = np.ones((x.shape[0], 1))
    X = np.hstack((x, colonna_1))
    z_1 = g(np.dot(W, X.T),sigma)
    Q = (1/P) * np.dot(z_1,z_1.T) + rho*np.eye(N)  
    return Q

def coeff(x, y, W, sigma): 
    P=x.shape[0]
    colonna_1 = np.ones((x.shape[0], 1))
    X = np.hstack((x, colonna_1))
    z_1 = g(np.dot(W, X.T),sigma)
    c = ((1/P) * np.dot(y.T,z_1.T)).T
    return c


def train_mlp_unsupervised(x, y, N, rho, sigma,seme):
    P=x.shape[0]  
    n_inputs = x.shape[1] 
    np.random.seed(seme)
    W = np.random.randn(N, n_inputs + 1)     
    Q = hessE(x, W, rho, sigma, N)
    c = coeff(x, y, W, sigma)
    V_star, residuals, rank, eigenvectors = np.linalg.lstsq(Q,c,rcond=None)
    
    return V_star.T , W



def grafico (W, V_opt, sigma, filename):
    x1_grid = np.linspace(-2, 2, 100)
    x2_grid = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    y_pred_grid = predict(grid_points, W, V_opt, sigma) 
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




