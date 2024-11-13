import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import glob
from tqdm import tqdm
from einops import rearrange
import os

TRAIN_VAL_SPLIT_RATIO = 0.8
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from setup import N_TILE_TYPES
BOARD_CHANNELS = N_TILE_TYPES+2 # env + center + empty

def boards_encoding(boards, device='cpu'):
    # boards : batch_size, 2 (tile type and crown), 5, 5
    boards = torch.as_tensor(boards, device=device, dtype=torch.int64)
    batch_size = boards.shape[0]
    board_size = boards.shape[-1]
    boards_one_hot = torch.zeros([
        batch_size,
        BOARD_CHANNELS+1, # (env + center + empty) + crowns 
        board_size,
        board_size],
        dtype=torch.int8,
        device=device)
    boards_one_hot.scatter_(1, (boards[:,0]+2).unsqueeze(1), 1)
    boards_one_hot[:,-1,:,:] = boards[:,1] # Place crowns at the end
    return boards_one_hot[:,1:]

class BoardsDataset(Dataset):
    def __init__(self, boards_dir, scores_dir, max_files):
        self.boards_files = sorted(glob.glob(f"{boards_dir}/boards_*.npy"))
        self.scores_files = sorted(glob.glob(f"{scores_dir}/scores_*.npy"))
        self.data = []
        self.max_files = max_files
        
        for i, (board_file,score_file) in tqdm(enumerate(zip(self.boards_files, self.scores_files)), total=self.max_files):
            if i > self.max_files:
                break
            boards = np.load(board_file)
            scores = np.load(score_file)
            
            num_samples, num_turns, num_players, channels, height, width = boards.shape
            boards = boards.reshape(-1, channels, height, width)

            scores = scores.reshape(-1)
            
            self.data.extend(zip(boards_encoding(boards, device=DEVICE), scores))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        boards, scores = self.data[idx]
        boards = boards.to(device=DEVICE)
        scores = torch.tensor(scores, dtype=torch.float32, device=DEVICE)
        return boards.to(torch.float32), scores
    
class ScorePredictorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv_2 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv_3 = nn.Conv2d(16, 16, kernel_size=3)
        
        # Calculate the output of the convolutional layers
        self.fc_layers = nn.Sequential(
            nn.Linear(400, 128),
            nn.SELU(),
            nn.Linear(128,128),
            nn.SELU(),
            nn.Linear(128,1)
        )
        
    def forward(self, x):
        # x shape : (batch_size, 8, 5, 5)
        pad = 1
        x = F.pad(x, (pad,)*4, "constant", -1)
        x = self.conv_1(x)
        x = F.relu(x)
        x = F.pad(x, (pad,)*4, "constant", -1)
        x = self.conv_2(x)
        x = F.relu(x)
        x = F.pad(x, (pad,)*4, "constant", -1)
        x = self.conv_3(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x.squeeze()
    
class IterativeScorePredictorCNN(nn.Module):
    def __init__(self, iters):
        super().__init__()
        
        self.iters = iters
        self.conv = nn.Conv2d(8, 8, kernel_size=3)
        
        # Calculate the output of the convolutional layers
        self.fc_layers = nn.Sequential(
            nn.Linear(200, 128),
            nn.SELU(),
            nn.Linear(128,128),
            nn.SELU(),
            nn.Linear(128,1)
        )
        
    def forward(self, x):
        # x shape : (batch_size, 8, 5, 5)
        for i in range(self.iters):
            x = F.pad(x, (1,1,1,1), "constant", -1)
            x = self.conv(x)
            x = F.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x.squeeze()
    
class FCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(BOARD_CHANNELS * 5 * 5, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 1))
        
    def forward(self, x):
        x = rearrange(x, 'b c w h -> b (c w h)')
        return self.fc_layers(x)

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, loss_save_folder='', loss_save_name='loss.npy'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(DEVICE)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        
        # Training loop
        for boards, scores in tqdm(train_loader):
            boards, scores = boards.to(DEVICE), scores.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * boards.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for boards, scores in val_loader:
                boards, scores = boards.to(DEVICE), scores.to(DEVICE)
                outputs = model(boards)
                val_loss = criterion(outputs, scores)
                running_val_loss += val_loss.item() * boards.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Save the best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
        # Save train and validation losses
    os.makedirs(loss_save_folder, exist_ok=True)
    np.save(os.path.join(loss_save_folder, 'train_' + loss_save_name), train_losses)
    np.save(os.path.join(loss_save_folder, 'val_' + loss_save_name), val_losses)
    
    # Load the best model weights
    model.load_state_dict(best_model_state)
    return model  # Return the best model

def save_model(model, save_path='predict_reward/models/score_predictor.pth'):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

#%%

# Initialize Dataset, DataLoader, and Model
dataset = BoardsDataset('predict_reward/data', 'predict_reward/data', 150)
train_size = int(TRAIN_VAL_SPLIT_RATIO * len(dataset))
val_size = int(len(dataset)) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#%%

# Train the model
# iters = 5
# model = IterativeScorePredictorCNN(iters=iters)
model = ScorePredictorCNN()
trained_model = train_model(model, train_loader, val_loader, 10, loss_save_folder='predict_reward/models', loss_save_name=f'conv_loss.npy')

# Save the trained model
save_model(trained_model, save_path='predict_reward/models/conv.pth')

#%%

max_loss = 0
criterion = nn.MSELoss()
model.eval()
running_val_loss = 0.0
with torch.no_grad():
    for boards, scores in val_loader:
        boards, scores = boards.to(DEVICE), scores.to(DEVICE)
        outputs = model(boards)
        val_loss = criterion(outputs, scores)
        idx = torch.abs(outputs.to(dtype=int) - scores) > 2
        print(boards[idx])

        running_val_loss += val_loss.item() * boards.size(0)

epoch_val_loss = running_val_loss / len(val_loader.dataset)