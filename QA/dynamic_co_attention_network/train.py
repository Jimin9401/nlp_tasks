import torch
from visualization import visualize

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model,iterator,optimizer,criterion,train_loader):
    model=model.to(device)
    model=model.train()
    for epoch in range(iterator):
        for k, (q_tensor, c_tensor,s_ix,e_ix) in enumerate(train_loader):
            q_tensor= torch.LongTensor(q_tensor).to(device)
            c_tensor= torch.LongTensor(c_tensor).to(device)
            s_ix=s_ix.to(device)
            e_ix=e_ix.to(device)

            optimizer.zero_grad()
            entropies= model.forward(q_tensor,c_tensor,train=True)

            loss_start=0
            loss_end=0

            for s,e in entropies:
                loss_start += criterion(s,s_ix)
                loss_end +=criterion(e,e_ix)
            loss=loss_start+loss_end
            loss.backward()
            optimizer.step()

            print("Epoch: {:d} batch step: [{:d}/{:d}] Loss: {:.4f}".format(epoch + 1, k + 1, len(train_loader), loss))
