import os
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
import pandas as pd
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix as skl_confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


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

    indexLabel7 = np.where((y_all_labels==7))
    xLabel7 =  X_all_labels[indexLabel7][:1000,:].astype('float64')
    yLabel7 = y_all_labels[indexLabel7][:1000].astype('float64')

    """
    To train a SVM in case of binary classification you have to convert the labels of the two classes of interest into '+1' and '-1'.
    """
    X_data = np.concatenate((xLabel1, xLabel5), axis=0)
    Y_data = np.concatenate((yLabel1, yLabel5), axis=0)
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

def decomposition_algorithm_MVP(X_train, Y_train, gamma, C):
    alpha = np.zeros(len(Y_train))
    alpha_old = np.zeros(len(Y_train))
    grad_f = - np.ones(len(Y_train))  
    d=np.zeros(2)   
    k = 0
    tol=1e-6
    Q_red_diz={} #memorizzo in un dizionario le colonne di Q già calcolate utilizzando come chiave l'indice della colonna
    number_of_Q_rows_computed=0
    m=1
    M=-1
    while m > M+10**-8:
        grad_dot_y = - grad_f*Y_train
        indici_R = np.where(((alpha < C - tol) & (Y_train > 0)) | ((alpha > 0 + tol) & (Y_train < 0)))[0]
        indici_S = np.where(((alpha < C - tol) & (Y_train < 0)) | ((alpha > 0 + tol) & (Y_train > 0)))[0]
        I =indici_R[np.argmax(grad_dot_y[indici_R])]  
        J =indici_S[np.argmin(grad_dot_y[indici_S])] 
        m = grad_dot_y[indici_R[np.argmax(grad_dot_y[indici_R])] ]
        M = grad_dot_y[indici_S[np.argmin(grad_dot_y[indici_S])] ]
        
        KKT_violation=m-M
        #print(k,"diff = ",(m-M))      
        W = np.array([I, J]) 
        key1=I
        key2=J 
        

        #NB SONO STATI TESTATI DUE METODI PER MEMORIZZARE LE RIGHE DI Q IN UN DIZIONARIO: 
    
        # IL PRIMO METODO CALCOLA LA RIGA DI Q NECESSARIA SINGOLARMENTE E LA MEMORIZZA NEL DIZIONARIO.
        # METODO 1 (1.10 secondi) ------------------------------------------------------------------------------------------------------------------------------------------
        '''
        if key1 not in Q_red_diz:
            Q_red = (( (Y_train.reshape(-1, 1) * rbf_kernel(X_train[[W[0]],:],X_train, gamma) ) *  Y_train[key1]).astype("float")).T
            Q_red_diz[key1] = Q_red  # Memorizzazo la riga nel dizionario
            number_of_Q_rows_computed+=1
        if key2 not in Q_red_diz:
            Q_red = (( (Y_train.reshape(-1, 1) * rbf_kernel(X_train[[W[1]],:],X_train, gamma) ) * Y_train[key2]).astype("float")).T 
            number_of_Q_rows_computed+=1
            Q_red_diz[key2] = Q_red  # Memorizzazo la riga nel dizionario
        Q_red = np.vstack((Q_red_diz[key1] , Q_red_diz[key2]))
        Q_W = Q_red[:, W]
        '''
        # ------------------------------------------------------------------------------------------------------------------------------------------

        # IL SECONDO METODO CALCOLA ENTRAMBE LE RIGHE DI Q RELATIVE AL WORKING SET TUTTE LE VOLTE CHE UNA DELLE 2 RIGHE NON è PRESENTE NEL DIZIONARIO.
        #METODO 2 (0.88 secondi) ------------------------------------------------------------------------------------------------------------------------------------------
        if key1 not in Q_red_diz:
            #Q_red = np.dot(np.dot(Y_diag,rbf_kernel(X_train[W,:],X_train, gamma)), Y_diag_W).astype("float") #questa va meno veloce, meglio quella sotto
            Q_red = (np.vstack((Y_train,Y_train)).T*rbf_kernel(X_train[W,:],X_train, gamma)* Y_train[W]).astype("float") 
            number_of_Q_rows_computed+=2
            Q_red_diz[key1] = Q_red[:,0]  # Memorizzazo la riga nel dizionario
            if key2 not in Q_red_diz:
                Q_red_diz[key2] = Q_red[:,1]
            Q_red = np.vstack((Q_red[:,0] , Q_red[:,1]))
        elif key2 not in Q_red_diz:
            #Q_red = np.dot(np.dot(Y_diag,rbf_kernel(X_train[W,:],X_train, gamma)), Y_diag_W).astype("float") #questa va meno veloce, meglio quella sotto
            Q_red = (np.vstack((Y_train,Y_train)).T*rbf_kernel(X_train[W,:],X_train, gamma)* Y_train[W]).astype("float") 
            Q_red_diz[key2] = Q_red[:,1]  # Memorizzazo la riga nel dizionario
            Q_red = np.vstack((Q_red[:,0] , Q_red[:,1]))
            number_of_Q_rows_computed+=2
        else:
            Q_red = np.vstack((Q_red_diz[key1] , Q_red_diz[key2]))
        Q_W = Q_red[:, W]
        # ------------------------------------------------------------------------------------------------------------------------------------------

        Y_train_W=Y_train[W]
        d[0]=Y_train_W[0] #imposto la prima componente all'indice i appartenente a R (che si trova nella prima componente di W) di Y_train. N.B.
        d[1]=-Y_train_W[1] #imposto la prima componente all'indice j appartenente a S (che si trova nella seconda componente di W) di Y_train. N.B.

        d_Q_d=np.dot(np.dot(d.T,Q_W),d)
       
        if d_Q_d > 0:
            t_star= - (np.dot(grad_f[W].T,d))/d_Q_d
        else:
            t_star=np.inf 
        t_max_amm_pos = np.where(d > 0, (C - alpha[W]) / d, np.inf) 
        t_max_amm_neg = np.where(d < 0, alpha[W] / -d, np.inf) 
        t_max_amm = np.min([t_max_amm_pos, t_max_amm_neg])
        t = min(t_star, t_max_amm)
        alpha_W_old=alpha[W]   
        alpha[W] = alpha_W_old + t * d   
        grad_f += np.dot((np.array(Q_red).T),(alpha[W]-alpha_old[W]).T) 
        alpha_old = np.copy(alpha)     
        k=k+1
    
    return alpha, number_of_Q_rows_computed, k, KKT_violation


def dual_objective(K_train, Y_train, alpha): 
    Y_diag = np.diag(Y_train)
    Q = np.dot(np.dot(Y_diag,K_train), Y_diag)
    e = np.ones(len(Y_train))
    return  0.5*np.dot(np.dot(alpha.T, Q),alpha) - np.dot(e.T,alpha)




def predict(Y_train,K_train, K_test, alpha, C):
    indici_SV = np.where((alpha > 1e-6) & (alpha < C - 1e-6))[0]
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

