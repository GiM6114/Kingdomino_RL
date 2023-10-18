import torch
import torch.nn as nn
from torchviz import make_dot
from IPython.display import display

class NN(nn.Module):
    def __init__(self, inpt, outpt):
        super().__init__()
        self.fc = nn.Linear(
            in_features  = inpt,
            out_features = outpt)
    def forward(self, x):
        return self.fc(x)

class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn_shared = NN(2,2)
        self.nn_post = NN(4,1)
        
    def forward(self, x):
        first_half_outpt = self.nn_shared(x[:,0])
        second_half_outpt = self.nn_shared(x[:,1])
        x = torch.cat((first_half_outpt, second_half_outpt), dim=1)
        x = self.nn_post(x)
        return x
        
#%% Works      

X = torch.rand(size=(3,2,2))
y = (X.transpose(1,2) @ torch.tensor([2.,1.])).sum(dim=1)

model = Shared()
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                              amsgrad=True,
                              lr=0.01)

for i in range(10):
    y_hat = model(X).squeeze()
    loss = loss_fn(y, y_hat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#%% ERROR RECREATED

X = torch.rand(size=(3,2,2))
y = (X.transpose(1,2) @ torch.tensor([2.,1.])).sum(dim=1)
y_1 = y * 2
y_2 = y * 0.5

shared = Shared()
model_1 = NN(1,1)
model_2 = NN(1,1)
loss_fn_1 = nn.MSELoss()
loss_fn_2 = nn.MSELoss()
optimizer_1 = torch.optim.AdamW(
    list(shared.parameters()) + list(model_1.parameters()),
    amsgrad=True,
    lr=0.01)
optimizer_2 = torch.optim.AdamW(
    list(shared.parameters()) + list(model_2.parameters()),
    amsgrad=True,
    lr=0.01)

for i in range(10):
    y_hat = shared(X)
    
    y_hat_1 = model_1(y_hat)
    loss_1 = loss_fn_1(y_1, y_hat_1)
    optimizer_1.zero_grad()
    loss_1.backward(retain_graph=True)
    optimizer_1.step()
    
    graph = make_dot(
        model_2(y_hat), 
        params=dict(shared.named_parameters()) | dict(model_2.named_parameters()),)
        # show_attrs=True,
        # show_saved=True)
    display(graph)

    y_hat_2 = model_2(y_hat)
    loss_2 = loss_fn_2(y_2, y_hat_2)
    optimizer_2.zero_grad()
    loss_2.backward()
    optimizer_2.step()

#%% FIX #1 (NOT ENTIRELY SATISFACTORY) :
    # In the comp graph cut the second network from the shared network
    # --> updates on the shared network only come from the first network

X = torch.rand(size=(3,2,2))
y = (X.transpose(1,2) @ torch.tensor([2.,1.])).sum(dim=1)
y_1 = y * 1.5
y_2 = y * 0.5

shared = Shared()
model_1 = NN(1,1)
model_2 = NN(1,1)
loss_fn_1 = nn.MSELoss()
loss_fn_2 = nn.MSELoss()
optimizer_1 = torch.optim.AdamW(
    list(shared.parameters()) + list(model_1.parameters()),
    amsgrad=True,
    lr=0.01)
optimizer_2 = torch.optim.AdamW(model_2.parameters(),
    amsgrad=True,
    lr=0.01)

for i in range(5000):
    y_hat_shared_1 = shared(X)
    
    y_hat_1 = model_1(y_hat_shared_1).squeeze()

    loss_1 = loss_fn_1(y_1, y_hat_1)
    optimizer_1.zero_grad()
    loss_1.backward()
    optimizer_1.step()
    
    with torch.no_grad():
        y_hat_shared_2 = y_hat_shared_1.clone()
    y_hat_2 = model_2(y_hat_shared_2).squeeze()
    loss_2 = loss_fn_2(y_2, y_hat_2)
    optimizer_2.zero_grad()
    loss_2.backward()
    optimizer_2.step()
    
#%% FIX #2 SIMPLY RECOMPUTE SHARED ??? HOW DID I NOT SEE THAT EARLIER

X = torch.rand(size=(3,2,2))
y = (X.transpose(1,2) @ torch.tensor([2.,1.])).sum(dim=1)
y_1 = y * 1.5
y_2 = y * 0.5

shared = Shared()
model_1 = NN(1,1)
model_2 = NN(1,1)
loss_fn_1 = nn.MSELoss()
loss_fn_2 = nn.MSELoss()
optimizer_1 = torch.optim.AdamW(
    list(shared.parameters()) + list(model_1.parameters()),
    amsgrad=True,
    lr=0.01)
optimizer_2 = torch.optim.AdamW(
    list(shared.parameters()) + list(model_2.parameters()),
    amsgrad=True,
    lr=0.01)

for i in range(5000):
    y_hat_shared_1 = shared(X)
    
    y_hat_1 = model_1(y_hat_shared_1).squeeze()

    loss_1 = loss_fn_1(y_1, y_hat_1)
    optimizer_1.zero_grad()
    loss_1.backward(retain_graph=True)
    optimizer_1.step()
    
    y_hat_shared_2 = shared(X)
    y_hat_2 = model_2(y_hat_shared_2).squeeze()
    loss_2 = loss_fn_2(y_2, y_hat_2)
    optimizer_2.zero_grad()
    loss_2.backward()
    optimizer_2.step()