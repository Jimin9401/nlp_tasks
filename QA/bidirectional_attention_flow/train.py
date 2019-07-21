import torch
from visualization import visualize

if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"

def train(model,iterator,optimizer,criterion,train_loader):

    for epoch in range(iterator):
        for k, (q_tensor, c_tensor,s_ix,e_ix) in enumerate(train_loader):
            q_tensor= torch.LongTensor(q_tensor).to(device)
            c_tensor= torch.LongTensor(c_tensor).to(device)

            optimizer.zero_grad()
            p1,p2= model.forward(q_tensor,c_tensor)

            loss = criterion(p1,s_ix)+criterion(p2,e_ix)

            loss.backward()

            optimizer.step()

            print("Epoch: {:d} batch step: [{:d}/{:d}] Loss: {:.4f}".format(epoch + 1, k + 1, len(train_loader), loss))
