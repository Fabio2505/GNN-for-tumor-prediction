import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim  
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset  
import networkx as nx
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.utils import from_networkx
from sklearn.manifold import TSNE
from torch_geometric.data import Data


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

#visualize è usata in linea 189 per gli embeddings prima del training e dopo (linea 264) ma disabilitata da un "#"


def encode_labels(labels):
    labels = [label.strip() for label in labels] #togli spazi
    label_map = {"Luminal A": 0, "Luminal B": 1}  
    return [label_map[label] for label in labels]




#____________________ CORE ______________________________________________________________#

#percorso="\\dataset_LUMINAL_A_B.csv"
#dataframe=pd.read_csv(percorso,sep=',')
dataframe=pd.read_csv("dataset_LUMINAL_A_B.csv",sep=',')


#print(dataframe.shape) (100, 1023). l'indice del df non conta


X=dataframe.drop(columns=['l'])
y=dataframe['l'] 

scaler_minmax = MinMaxScaler(feature_range=(0, 1))
X= scaler_minmax.fit_transform(X)  
X=pd.DataFrame(X)
correlation_matrix = X.T.corr(method='pearson').values
print(f"correlation matrix shape: {correlation_matrix.shape}")
#print() 
np.fill_diagonal(correlation_matrix, 0)
y=encode_labels(y) #converti in numero


#print(type(correlation_matrix))
#print(correlation_matrix.shape)
#print(np.isnan(correlation_matrix).any())
mask = abs(correlation_matrix) > 0.2
correlation_matrix[~mask] = 0


G = nx.from_numpy_array(correlation_matrix)

data = Data(
    x=torch.tensor(X.values, dtype=torch.float),  # Caratteristiche dei nodi
    edge_index=torch.tensor(list(G.edges), dtype=torch.long).t().contiguous(),  # Indici degli archi (Grafo filtrato), contiguous sistema in memoria il vettore trasposto
    y=torch.tensor(y, dtype=torch.long)  # Etichette dei nodi
)

# Numero di nodi e archi
print(f"Numero di nodi: {G.number_of_nodes()}")
print(f"Numero di archi: {G.number_of_edges()}")
print("Indici dei nodi:", list(G.nodes))





# Trasponi se necessario
if X.shape[0] != len(G.nodes):  # Se il numero di righe non corrisponde ai nodi
    X = X.T  # Trasponi la matrice delle feature pandas

data.y = torch.tensor(y, dtype=torch.long)  #etichette dei nodi

# Controlla i dettagli del dataset
print(data)



#colori per le due classi
color_map = {0: 'blue', 1: 'red'} #blue = luminal A

for i, label in enumerate(y):
    G.nodes[i]['label'] = label

node_colors = [color_map[G.nodes[i]['label']] for i in G.nodes]

# Disposizione dei nodi
pos = nx.spring_layout(G)

# Creazione della figura
plt.figure(figsize=(8, 6))

# nodi con colore della classe
nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)

# archi
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

#etichette dei nodi 
node_labels = {i: G.nodes[i]['label'] for i in G.nodes}
#nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color="black")


plt.title("Grafo con Nodi Colorati per Due Classi")
plt.axis("off")
#plt.show()
'''
# Diverse disposizioni dei nodi
layouts = {
    "Spring Layout": nx.spring_layout(G),
    "Circular Layout": nx.circular_layout(G),
    "Kamada Kawai Layout": nx.kamada_kawai_layout(G),
    "Spectral Layout": nx.spectral_layout(G),
    "Shell Layout": nx.shell_layout(G),
}

# Visualizzazione dei diversi layout
plt.figure(figsize=(16, 12))

for i, (layout_name, layout_pos) in enumerate(layouts.items(), start=1):
    plt.subplot(2, 3, i)
    nx.draw_networkx_nodes(G, layout_pos, node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, layout_pos, width=1.0, alpha=0.5)
    plt.title(layout_name)
    plt.axis("off")

plt.tight_layout()
#plt.show()
'''


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
    
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.dropout = dropout 
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout 
        self.conv3 = GCNConv(hidden_dim, output_dim)
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training) 
        x = self.conv3(x, edge_index)
         
        return F.log_softmax(x, dim=1)


# Parametri del modello
input_dim = 1022
hidden_dim = 10
output_dim = 2
k_folds = 5
num_epochs = 100
learning_rate = 0.01



model = GCN(1022,10,2,0.3)
model.eval()
out = model(data)
#visualize(out, color=data.y)         



# Liste per metriche
train_losses = []
val_losses = []
accuracies = []
f1_scores = []
random_seeds = [7, 42, 123, 99]

final_metric=[] #mean accuracy across seeds

for seed in random_seeds:
    print(f'\n🔄 Running with random seed: {seed}')
    
   
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Creare i fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=2024)
    node_indices = np.arange(data.num_nodes)

    for fold, (train_idx, test_idx) in enumerate(kf.split(node_indices)):
        print(f'\nFold {fold+1}/{k_folds}')
        
        # Creare maschere di train e test
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        
        # Inizializzazione modello
        model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Reset delle perdite per il fold corrente
        fold_train_losses = []
        fold_val_losses = []
        
        # Training Loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            
            train_loss = loss_fn(out[train_mask], data.y[train_mask])
            train_loss.backward()
            optimizer.step()
            
            fold_train_losses.append(train_loss.item())
            
            # validazione
            model.eval()
            
            

            with torch.no_grad():
                val_out = model(data)
                val_loss = loss_fn(val_out[test_mask], data.y[test_mask])
                fold_val_losses.append(val_loss.item())
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        # Loss
        mean_fold_train_loss = np.mean(fold_train_losses)
        mean_fold_val_loss = np.mean(fold_val_losses)
        train_losses.append(mean_fold_train_loss)
        val_losses.append(mean_fold_val_loss)
        
    
        model.eval()
        #visualize(out, color=data.y)              #Visualizza embeddings post training
        preds = model(data).argmax(dim=1)
        
        # Accuracy
        accuracy = (preds[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
        accuracies.append(accuracy)
        
        # F1-Score
        y_true = data.y[test_mask].cpu().numpy()
        y_pred = preds[test_mask].cpu().numpy()
        f1 = f1_score(y_true, y_pred, average='weighted')
        f1_scores.append(f1)
        
        print(f'Fold {fold+1} - Train Loss: {mean_fold_train_loss:.4f}, Val Loss: {mean_fold_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}')

    # Media e deviazione standard delle metriche
    mean_train_loss = np.mean(train_losses)
    mean_val_loss = np.mean(val_losses)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_f1_score = np.mean(f1_scores)

    # Risultati Finali
    #print(f'\nFinal K-Fold Results:')
    #print(f'Mean Training Loss: {mean_train_loss:.4f}')
    #print(f'Mean Validation Loss: {mean_val_loss:.4f}')
    print(f'Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
    print(f'Mean F1-Score: {mean_f1_score:.4f}')
    final_metric.append(mean_accuracy)

final_metric=np.array(final_metric, dtype=float)
print(final_metric.mean())