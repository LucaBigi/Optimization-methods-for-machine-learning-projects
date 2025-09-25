import os
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import pandas as pd
import time
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

def decomposition_algorithm(X_train, Y_train, q, gamma, C):
    alpha = np.zeros(len(Y_train))
    alpha_old = np.zeros(len(Y_train))
    grad_f = - np.ones(len(Y_train))     
    k = 0
    n_tot_iter=0
    tol=1e-6
    Q_red_diz = {}  # Dizionario per memorizzare le righe di Q calcolate 
    sol_status_list=[]
    while True:
        grad_dot_y = - grad_f*Y_train
        indici_R = np.where(((alpha < C - tol) & (Y_train > 0)) | ((alpha > 0 + tol) & (Y_train < 0)))[0]
        indici_S = np.where(((alpha < C - tol) & (Y_train < 0)) | ((alpha > 0 + tol) & (Y_train > 0)))[0]
        I =indici_R[np.argsort(-grad_dot_y[indici_R])] #np.argsort ordina -grad_dot_y[indici_R] in ordine crescente e restituisce gli indici corrispondenti
        J =indici_S[np.argsort(grad_dot_y[indici_S])] #np.argsort ordina grad_dot_y[indici_S] in ordine crescente e restituisce gli indici corrispondenti
        m = grad_dot_y[I[0]]
        M = grad_dot_y[J[0]] 
        #print(k,"diff = ",(m-M))
        if m <= M+10**-7:
            KKT_violation=m-M
            break
        W = np.concatenate([I[:q//2], J[:q//2]]) #concatena in un unico array gli ultimi q/2 indici di I e i primi q/2 indici di J 
        
        
        #METODO CON DIZIONARIO --------------------------------------------------------------------------------------------------
        #PIU' EFFICENTE --> CALCOLO INSIEME TUTTE LE RIGHE DI Q CHE NON SONO NEL DIZIONARIO E POI CON UN CICLO FOR ACCEDO AD OGNI CHIAVE DEL DIZIONARIO E MEMORIZZO LE RIGHE. SUBITO DOPO POPOLO LE MATRICI DI CUI HO BISOGNO CON I VALORI DI Q NECESSARI
        W_negato = np.setdiff1d(np.arange(len(Y_train)), W)
        P_W = len(W)
        Q = np.zeros((len(W), len(W)))
        Q_p = np.zeros((len(W_negato), len(W)))
        Q_1 = np.zeros((len(Y_train), len(W)))
        needed_Q_rows = np.setdiff1d(W, np.array(list(Q_red_diz.keys())))

        if len(needed_Q_rows)!=0:
            Q_red = (( (Y_train.reshape(-1, 1) * rbf_kernel(X_train[needed_Q_rows,:],X_train, gamma) ) *  Y_train[needed_Q_rows]).astype("float")).T
        for i, index in enumerate(W):
            if index in needed_Q_rows:
                idx=np.where(index==needed_Q_rows)[0]
                Q_red_diz[index] = Q_red[idx, :].reshape(1,-1)
            Q[i,:] = (Q_red_diz[index])[:,W].flatten()
            Q_p[:,i] = (Q_red_diz[index])[:,W_negato].flatten()
            Q_1[:,i] = (Q_red_diz[index]).flatten()    
        Q=Q.astype("float") 
        p = (np.dot(alpha[W_negato], Q_p) - np.ones(P_W)).astype("float") 



        #METODO SENZA DIZIONARIO  ---------------------------------------------------------------------------------
        #K_W_all=rbf_kernel(X_train[W,:],X_train, gamma) #calcolandolo qui lo calcolo una volta sola per ogni ciclo    
        #W_negato = np.setdiff1d(np.arange(len(Y_train)), W)
        #K_train_W = K_W_all[W,:]
        #Y_train_W = Y_train[W]
        #Y_train_W_negato = Y_train[W_negato]
        #alpha_W_negato = alpha[W_negato]
        #P_W = len(Y_train_W)
        #Y_diag_W = np.diag(Y_train_W)
        #Y_diag_W_negato=np.diag(Y_train_W_negato)
        #K_W_W_negato=K_W_all[W_negato,:]
        #Q = np.dot(np.dot(Y_diag_W,K_train_W), Y_diag_W).astype("float") 
        #p = (np.dot(alpha[W_negato], np.dot(np.dot(Y_diag_W_negato,K_W_W_negato), Y_diag_W)) - np.ones(P_W)).astype("float") 
        #Q_1 = np.dot(np.dot(np.diag(Y_train),K_W_all), Y_diag_W).astype("float") 




        # creo matrici per i vincoli 0 <= alpha_i <= C 
        G = np.vstack([-np.eye(P_W), np.eye(P_W)]).astype("float")   # G = [-I  I].T
        h = np.hstack([np.zeros(P_W), C * np.ones(P_W)]).astype("float")    # h = [0 ... 0  C ... C].T
        # creo parametri per i vincoli Y * alpha = b
        A = Y_train[W].reshape(1, -1).astype("float") 
        b = -np.dot(alpha[W_negato],Y_train[W_negato].T).astype("float") 
        Q = matrix(Q)
        p = matrix(p)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        solvers.options['show_progress']=False #False serve per nascondere il progresso ottimizzazione
        solvers.options['abstol'] = 1e-12
        solvers.options['reltol'] = 1e-13
        solvers.options['feastol'] = 1e-13
        sol = solvers.qp(Q,p, G, h, A, b)
        alpha_W = (np.array(sol['x']).flatten())
        alpha[W] = alpha_W
        grad_f += np.dot((np.array(Q_1)),(alpha_W-alpha_old[W]).T) 
        alpha_old = np.copy(alpha)     
        k=k+1
        n_tot_iter+=sol['iterations']
        if sol['status'] not in sol_status_list:
            sol_status_list.append(sol['status'])
    return alpha, k, n_tot_iter, KKT_violation, sol_status_list



def dual_objective(K_train, Y_train, alpha): 
    Y_diag = np.diag(Y_train)
    Q = np.dot(np.dot(Y_diag,K_train), Y_diag)
    e = np.ones(len(Y_train))
    #alpha=np.where(alpha<1e-6, 0, alpha)
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


# Function to plot the 4 graphs based on the lists provided
def plot_q_variation(error_rate_list, optimization_time_list, n_subproblems_list, KKT_viol_list):
    # Create a figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # First plot: Error rate vs q
    axs[0, 0].plot(range(4, 90, 2), error_rate_list, label="Error Rate", color='tab:red')
    axs[0, 0].set_title("Error Rate vs q")
    axs[0, 0].set_xlabel("q (working set size)")
    axs[0, 0].set_ylabel("Error Rate")
    axs[0, 0].grid(True)

    # Second plot: Optimization time vs q
    axs[0, 1].plot(range(4, 90, 2), optimization_time_list, label="Optimization Time", color='tab:blue')
    axs[0, 1].set_title("Optimization Time vs q")
    axs[0, 1].set_xlabel("q (working set size)")
    axs[0, 1].set_ylabel("Optimization Time (seconds)")
    axs[0, 1].grid(True)

    # Third plot: Number of subproblems solved vs q
    axs[1, 0].plot(range(4, 90, 2), n_subproblems_list, label="Number of Subproblems", color='tab:green')
    axs[1, 0].set_title("Number of Subproblems vs q")
    axs[1, 0].set_xlabel("q (working set size)")
    axs[1, 0].set_ylabel("Number of Subproblems")
    axs[1, 0].grid(True)

    # Fourth plot: KKT Violations vs q
    axs[1, 1].plot(range(4, 90, 2), KKT_viol_list, label="KKT Violations", color='tab:purple')
    axs[1, 1].set_title("KKT Violations vs q")
    axs[1, 1].set_xlabel("q (working set size)")
    axs[1, 1].set_ylabel("KKT Violations")
    axs[1, 1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


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



