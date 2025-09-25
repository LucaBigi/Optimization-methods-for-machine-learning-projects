import os
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
import pandas as pd
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import confusion_matrix as skl_confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


def load_dataset(path, kind='train'):

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    X_all_labels, y_all_labels = images, labels

    """
    We are only interested in the items with label 1, 5 and 7.
    Only a subset of 1000 samples per class will be used.
    """
    indexLabel1 = np.where((y_all_labels==1))
    xLabel1 =  X_all_labels[indexLabel1][:1000,:].astype('float64')
    yLabel1 = y_all_labels[indexLabel1][:1000].astype('float64')

    indexLabel5 = np.where((y_all_labels==5))
    xLabel5 =  X_all_labels[indexLabel5][:1000,:].astype('float64')
    yLabel5 = y_all_labels[indexLabel5][:1000].astype('float64')

    #indexLabel7 = np.where((y_all_labels==7))
    #xLabel7 =  X_all_labels[indexLabel7][:1000,:].astype('float64')
    #yLabel7 = y_all_labels[indexLabel7][:1000].astype('float64')

    """
    To train a SVM in case of binary classification you have to convert the labels of the two classes of interest into '+1' and '-1'.
    """
    X_data = np.concatenate((xLabel1, xLabel5), axis=0)
    Y_data = np.concatenate((yLabel1, yLabel5), axis=0)
    #Y_data= np.where(Y_data == 5, -1, Y_data) non lo faccio qua per garantire risultati analoghi nella question 4 quando considero solo le classi 1 e 5
    X_train, X_test, Y_train, Y_test=train_test_split(X_data, Y_data, test_size=0.2,shuffle=True, random_state=1817398, stratify=Y_data) # --> Stratified split preserva le proporzioni delle classi
    Y_train = np.where(Y_train == 5, -1, Y_train) #convert the label 5 in -1
    Y_test = np.where(Y_test == 5, -1, Y_test) #convert the label 5 in -1
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train) 
    X_test=scaler.transform(X_test) 
    
    return X_train, X_test, Y_train, Y_test


def rbf_kernel(X,X2, gamma):
    squared_norm = np.sum(X**2, axis=1) # sommo le componenti di ogni riga al quadrato e ottengo un vettore 1xP
    squared_norm2 = np.sum(X2**2, axis=1) # sommo le componenti di ogni riga al quadrato e ottengo un vettore 1xN
    K = np.exp(-gamma * (squared_norm.reshape(1, -1) + squared_norm2.reshape(-1, 1) - 2 * np.dot(X, X2.T).T)) #NxP <--  # 1xP + Nx1 + NxP 

    return K

def polynomial_kernel(X,X2, gamma):
    K = (np.dot(X, X2.T).T + 1) ** gamma 

    return K

def solve_svm_dual(K_train, Y_train, C):
    P = len(Y_train)
    Y_diag = np.diag(Y_train)
    Q = np.dot(np.dot(Y_diag,K_train), Y_diag).astype("float") 
    e = np.ones(P).astype("float") 
    # creo matrici per i vincoli 0 <= alpha_i <= C 
    G = np.vstack([-np.eye(P), np.eye(P)]).astype("float")   # G = [-I  I].T
    h = np.hstack([np.zeros(P), C * e]).astype("float")    # h = [0 ... 0  C ... C].T
    # creo parametri per i vincoli Y * alpha = b
    A = Y_train.reshape(1, -1).astype("float") 
    b = np.array([0.0]).astype("float") 

    Q = matrix(Q)
    p = matrix(-e)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    solvers.options['show_progress']=False #False serve per nascondere il progresso ottimizzazione
    solvers.options['abstol'] = 1e-9
    solvers.options['reltol'] = 1e-12
    solvers.options['feastol'] = 1e-8
    sol = solvers.qp(Q,p, G, h, A, b)
    alpha = (np.array(sol['x']).flatten())
    grad_f = np.dot(Q, alpha) - e
    grad_dot_y = - grad_f*Y_train
    tol=1e-6
    indici_R = np.where(((alpha < C - tol) & (Y_train > 0)) | ((alpha > 0 + tol) & (Y_train < 0)))[0]
    indici_S = np.where(((alpha < C - tol) & (Y_train < 0)) | ((alpha > 0 + tol) & (Y_train > 0)))[0]
    I =indici_R[np.argsort(-grad_dot_y[indici_R])] #np.argsort ordina -grad_dot_y[indici_R] in ordine crescente e restituisce gli indici corrispondenti
    J =indici_S[np.argsort(grad_dot_y[indici_S])] #np.argsort ordina grad_dot_y[indici_S] in ordine crescente e restituisce gli indici corrispondenti
    m = grad_dot_y[I[0]]
    M = grad_dot_y[J[0]] 
    KKT_violation=m-M

    return alpha, sol['iterations'], KKT_violation, sol['status']

def dual_objective(K_train, Y_train, alpha): 
    Y_diag = np.diag(Y_train)
    Q = np.dot(np.dot(Y_diag,K_train), Y_diag)
    e = np.ones(len(Y_train))
    return  0.5*np.dot(np.dot(alpha.T, Q),alpha) - np.dot(e.T,alpha)

def predict(Y_train,K_train, K_test, alpha, C):
    indici_SV = np.where((alpha > 1e-6) & (alpha < C - 1e-6))[0]
    #print("numero di free SV = ", len(indici_SV))
    #if len(indici_SV) == 0:
    #    raise ValueError("Nessun SV con 0 < alpha < C.")
    #indici_BSV = np.where((alpha >= C - 1e-6 ) & (alpha <= C + 1e-6))[0]
    #print("# BSV / # totale sample = ", indici_BSV.shape[0], "/", alpha.shape[0])
    K_sv = K_train[:,indici_SV]  # K_sv contiene solo le colonne di K associate a indici_SV
    Y_sv = Y_train[indici_SV]     # Y_sv contiene solo le etichette corrispondenti ai SV
    b_values = Y_sv - np.dot(alpha*Y_train, K_sv)
    b= np.mean(b_values)
    arg = np.dot(alpha * Y_train, K_test.T) + b 
    y_pred = np.sign(arg) #output (+1 o -1)
    return y_pred



def grid_search_with_k_fold_cross_validation(X, Y):
    minimo_validation=[1,1, None, None]
    error_rate_list=[]

    #prima grid search
    #lista_gamma=[0.1, 0.095, 0.09, 0.085, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005, 0.0001]
    #lista_C=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    
    #seconda grid search
    #lista_gamma=[0.0005, 0.00045, 0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.00015, 0.0001, 0.00005, 0.00001] # [0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0,0001,0.00011,0.00012,0.00013,0.00014,0.00015,0.00016,0.00017,0.00018,0.00019,0.00020]
    #lista_C=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148]

    # grid search per grafici finali
    #lista_gamma=[0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001,0.00011,0.00012,0.00013,0.00014,0.00015, 0.00016, 0.00017, 0.00018, 0.00019, 0.00020, 0.00021, 0.00022, 0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00029, 0.00030, 0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.00040, 0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00047, 0.00048, 0.00049, 0.00050, 0.00051, 0.00052, 0.00053, 0.00054, 0.00055, 0.00056, 0.00057, 0.00058, 0.00059, 0.00060, 0.00061, 0.00062, 0.00063, 0.00064, 0.00065, 0.00066, 0.00067, 0.00068, 0.00069, 0.00070, 0.00071, 0.00072, 0.00073, 0.00074, 0.00075, 0.00076, 0.00077, 0.00078, 0.00079, 0.00080, 0.00081, 0.00082, 0.00083, 0.00084, 0.00085, 0.00086, 0.00087, 0.00088, 0.00089, 0.00090, 0.00091, 0.00092, 0.00093, 0.00094, 0.00095, 0.00096, 0.00097, 0.00098, 0.00099, 0.001]
    #lista_C=[15]

    # grid search per grafici finali
    lista_gamma=[0.00008]
    lista_C=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9, 6.1, 6.3, 6.5, 6.7, 6.9, 7.1, 7.3, 7.5, 7.7, 7.9, 8.1, 8.3, 8.5, 8.7, 8.9, 9.1, 9.3, 9.5, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7, 11.9, 12.1, 12.3, 12.5, 12.7, 12.9, 13.1, 13.3, 13.5, 13.7, 13.9, 14.1, 14.3, 14.5, 14.7, 14.9, 15.1, 15.3, 15.5, 15.7, 15.9, 16.1, 16.3, 16.5, 16.7, 16.9, 17.1, 17.3, 17.5, 17.7, 17.9, 18.1, 18.3, 18.5, 18.7, 18.9, 19.1, 19.3, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5, 20.7, 20.9, 21.1, 21.3, 21.5, 21.7, 21.9, 22.1, 22.3, 22.5, 22.7, 22.9, 23.1, 23.3, 23.5, 23.7, 23.9, 24.1, 24.3, 24.5, 24.7, 24.9, 25.1, 25.3, 25.5, 25.7, 25.9, 26.1, 26.3, 26.5, 26.7, 26.9, 27.1, 27.3, 27.5, 27.7, 27.9, 28.1, 28.3, 28.5, 28.7, 28.9, 29.1, 29.3, 29.5, 29.7, 29.9, 30.1]


    k=5 #numero di fold per k-fold-cross-validation
    P = X.shape[0] 
    fold_size = P // k
    indices = np.arange(P) 
    for gamma in lista_gamma:
        for C in lista_C:
            print("gamma = ",gamma,"C = ", C)
            training_error_rate = []  # Lista per memorizzare l'error rate (e gli iperparametri usati) sui k fold del training set
            validation_error_rate = []  # Lista per memorizzare l'error rate (e gli iperparametri usati) sui k fold del validation set
            for i in range(k):
                validation_indices = indices[i * fold_size:(i + 1) * fold_size] 
                train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
                X_train = X[train_indices]
                Y_train = Y[train_indices]
                X_validation = X[validation_indices]
                Y_validation = Y[validation_indices]
                
                ##SCEGLI IL KERNEL:
                K_train=rbf_kernel(X_train,X_train, gamma) # NB: kernel rbf ---> VA SCELTO GAMMA > 0
                K_validation=rbf_kernel(X_train,X_validation, gamma) 
                #K_train=polynomial_kernel(X_train,X_train, gamma) # NB: kernel polinomiale ---> NB: VA SCELTO GAMMA > 1
                #K_validation=polynomial_kernel(X_train,X_validation, gamma) 

                alpha,_,_,_ = solve_svm_dual(K_train, Y_train, C)
                y_train=predict(Y_train,K_train, K_train, alpha, C)
                y_validation=predict(Y_train,K_train, K_validation, alpha, C)
                res_training=y_train*Y_train
                res_validation=y_validation*Y_validation
                training_error_rate.append(np.count_nonzero(res_training == -1)/np.size(res_training))
                validation_error_rate.append(np.count_nonzero(res_validation == -1)/np.size(res_validation))
            mean_training_error_rate = np.mean(training_error_rate)
            mean_validation_error_rate = np.mean(validation_error_rate)
            if mean_validation_error_rate < minimo_validation[1]:
                minimo_validation=[mean_training_error_rate, mean_validation_error_rate, gamma, C]
            error_rate_list.append([mean_training_error_rate, mean_validation_error_rate, gamma, C])

    #creo un excel con i risultati della grid search        
    headers = ["mean_training_error_rate", "mean_validation_error_rate", "gamma", "C"]
    df = pd.DataFrame(error_rate_list, columns=headers)
    df.to_excel("grid_search.xlsx", index=False, engine='openpyxl')
    print(f"File salvato come '{"grid_search.xlsx"}'.")
    #restituisco in output la combinazione di iperparametri con minimo error rate sul validation set
    return minimo_validation


def Grafici_grid_search(file_path):
    # Carica i dati da Excel
    df = pd.read_excel(file_path)
    #df = df[df['C'] < 50]
    

    # Trova la riga con il valore minimo di mean_validation_error_rate
    optimal_row = df.loc[df['mean_validation_error_rate'].idxmin()]
    optimal_gamma = optimal_row['gamma']
    optimal_C = optimal_row['C']

    # Grafici 2D
    # Grafico per mean_validation_error_rate e mean_training_error_rate in funzione di gamma con C fisso
    filtered_by_C = df[df['C'] == optimal_C]
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_by_C['gamma'], filtered_by_C['mean_validation_error_rate'], label='Validation Error Rate')
    plt.plot(filtered_by_C['gamma'], filtered_by_C['mean_training_error_rate'], label='Training Error Rate')
    plt.xlabel('Gamma')
    plt.ylabel('Mean error Rate')
    plt.title(f'Mean error Rates with C={optimal_C}')
    plt.xticks(np.arange(min(filtered_by_C['gamma']), max(filtered_by_C['gamma']) +0.00001,+0.0001)) #per prima grid--->#+ 0.01, 0.01))  # Setta gli intervalli di gamma
    plt.legend()
    plt.grid(True)
    plt.savefig('mean_error_rates_C_fissato.png')  # Salva il grafico come immagine
    plt.close()

    # Grafico per mean_validation_error_rate e mean_training_error_rate in funzione di C con gamma fisso
    filtered_by_gamma = df[df['gamma'] == optimal_gamma]
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_by_gamma['C'], filtered_by_gamma['mean_validation_error_rate'], label='Validation Error Rate')
    plt.plot(filtered_by_gamma['C'], filtered_by_gamma['mean_training_error_rate'], label='Training Error Rate')
    plt.xlabel('C')
    plt.ylabel('mean error Rate')
    plt.title(f'mean error Rates with Gamma={optimal_gamma}')
    plt.legend()
    plt.grid(True)
    plt.savefig('mean_error_rates_gamma_fissato.png')  # Salva il grafico come immagine
    plt.close()

    # Grafico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['gamma'], df['C'], df['mean_validation_error_rate'], c='r', marker='o', label='Validation Error Rate')
    ax.scatter(df['gamma'], df['C'], df['mean_training_error_rate'], c='b', marker='^', label='Training Error Rate')
    ax.set_xlabel('Gamma')
    ax.set_ylabel('C')
    ax.set_zlabel('Mean error Rate')
    ax.set_title('3D plot of Mean Error Rates')
    ax.legend()
    ax.view_init(elev=20, azim=134)  
    plt.savefig('Mean_error_rates_3D.png')
    plt.close()

    print("Grafici salvati nella cartella corrente")


def print_confusion_matrix(y_test, Y_test):
    # Converto -1 in 5
    y_test = np.where(y_test == -1, 5, y_test)
    Y_test = np.where(Y_test == -1, 5, Y_test)
    # Calcola la matrice di confusione usando la funzione corretta di sklearn
    cm = skl_confusion_matrix(Y_test, y_test)
    # Crea una mappa di colori personalizzata che va dal giallo al viola
    cmap = LinearSegmentedColormap.from_list("purple_yellow", ["purple", "yellow"])
    # Crea un grafico con seaborn heatmap per la matrice di confusione
    plt.figure(figsize=(8, 6))  # Imposta la dimensione della figura
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, 
                xticklabels=['1', '5'],
                yticklabels=['1', '5'])

    # Titolo della matrice di confusione
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Salva come immagine e mostra la matrice di confusione
    plt.savefig('confusion_matrix.png', bbox_inches='tight')  # Salva la matrice
    plt.show()  # Mostra la matrice




