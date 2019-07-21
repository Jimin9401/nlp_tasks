import torch
from visualization import visualize


torch.manual_seed(777)

if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"


def train(model,iterator,optimizer,criterion,train_loader,visual_path,trg2idx,savepath):
    loss_for_save=100
    for epoch in range(iterator):
        for k, (src_batch, trg_batch) in enumerate(train_loader):
            src_tensor = torch.LongTensor(src_batch).to(device)
            trg_tensor = torch.LongTensor(trg_batch).to(device)

            optimizer.zero_grad()

            outputs = model.forward(src=src_tensor, trg=trg_tensor, teacher_force=True)

            outputs = outputs[1:].contiguous().view(-1, outputs.shape[-1])
            trg_tensor = trg_tensor[1:].contiguous().view(-1)
            loss = criterion(outputs, trg_tensor)
            visualize(loss,epoch,visual_path,model,src_tensor,trg_tensor,trg2idx=trg2idx)
            loss.backward()

            optimizer.step()
            if(loss.item()<loss_for_save):
                loss_for_save=loss.item()
                torch.save(model.state_dict(),savepath)
                print("save model at Epoch {:d}".format(epoch+1))


            print("Epoch: {:d} batch step: [{:d}/{:d}] Loss: {:.4f}".format(epoch + 1, k + 1, len(train_loader), loss))


def eval(model,iterator,optimizer,criterion,train_loader,visual_path,trg2idx):
    for k, (src_batch, trg_batch) in enumerate(train_loader):
        src_tensor = torch.LongTensor(src_batch).to(device)
        trg_tensor = torch.LongTensor(trg_batch).to(device)

        optimizer.zero_grad()

        outputs = model.forward(src=src_tensor, trg=trg_tensor, teacher_force=False)

        outputs = outputs[1:].contiguous().view(-1, outputs.shape[-1])
        trg_tensor = trg_tensor[1:].contiguous().view(-1)



