import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim  
import torch.nn.functional as F 
from torch.utils.data import DataLoader,TensorDataset,Dataset
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score



def encode_labels(labels):
    labels = [label.strip() for label in labels] #togli spazi
    label_map = {"Luminal A": 0, "Luminal B": 1}  
    return [label_map[label] for label in labels]




#____________________ CORE ______________________________________________________________#

#percorso="\\dataset_LUMINAL_A_B.csv"
dataframe=pd.read_csv("dataset_LUMINAL_A_B.csv",sep=',')


#print(dataframe.shape) (100, 1023). l'indice del df non conta


X=dataframe.drop(columns=['l'])
y=dataframe['l'] 

scaler_minmax = MinMaxScaler(feature_range=(0, 1))
X= scaler_minmax.fit_transform(X)  
y=encode_labels(y) #converti in numero

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

#print(X_train.shape) (80, 1022)




class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = MyDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)


torch.manual_seed(42) #42


# modello MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(1022, 10)
        #self.dropout = nn.Dropout(p=0.3)  better without! used earlystopping instead
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        #x = self.dropout(x)                better without!
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return F.log_softmax(x, dim=1)


# Parametri 
input_dim = 1022
hidden_dim = 10
output_dim = 2
k_folds = 5
num_epochs = 30
learning_rate = 0.001
batch_size = 8
random_seeds = [7, 42, 123, 99]


# Dataset
X_tensor = torch.tensor(X, dtype=torch.float32) #.values
y_tensor = torch.tensor(y, dtype=torch.long)
dataset=TensorDataset(X_tensor, y_tensor)


results = []
final_metric=[] #mean accuracy across seeds



for seed in random_seeds:
    print(f'\n🔄 Running with random seed: {seed}')
    
    # Imposta il seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)


    train_losses = []
    val_losses = []
    accuracies = []
    f1_scores = []

    # K-Fold Cross Validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)): #kf.split produce le k train-test mask. (num_fold, (mask_tr,mask_test))
        print(f'\nFold {fold+1}/{k_folds}')
        
        # Sottodataset per training e testing
        train_subsampler = torch.utils.data.Subset(dataset, train_idx)
        test_subsampler = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=False)
        
        # Inizializzazione
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        fold_train_losses = []
        fold_val_losses = []
        
        # Training Loop
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            fold_train_losses.append(epoch_loss / len(train_loader))
            
            # Validazione
            model.eval()
    
            val_loss = 0.0
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, index = torch.max(outputs, 1)
                    y_true.extend(batch_labels.cpu().numpy())  #extend altrimenti aggiungerei il vettore [batch]
                    y_pred.extend(index.cpu().numpy())
            
            fold_val_losses.append(val_loss / len(test_loader))
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        

        # Calcolo delle metriche
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        train_losses.append(np.mean(fold_train_losses))
        val_losses.append(np.mean(fold_val_losses))
        accuracies.append(accuracy)
        f1_scores.append(f1)
       
        print(f'Fold {fold+1} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}')

   
    mean_train_loss = np.mean(train_losses)
    mean_val_loss = np.mean(val_losses)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_f1_score = np.mean(f1_scores)

    #print(f'\nFinal K-Fold Results:')
    #print(f'Mean Training Loss: {mean_train_loss:.4f}')
    #print(f'Mean Validation Loss: {mean_val_loss:.4f}')
    print(f'Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
    print(f'Mean F1-Score: {mean_f1_score:.4f}')
   
    final_metric.append(mean_accuracy)

final_metric=np.array(final_metric, dtype=float)
print(final_metric.mean())




