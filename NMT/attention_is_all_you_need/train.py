from eval import evaluate

###
import torch
import numpy as np

torch.manual_seed(777)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(model, iterator, optimizer, criterion, train_loader, teacher_force, device, test_loader):
    model = model.train()
    match_score = 0

    for epoch in range(iterator):
        average_loss = []
        for k, (src_batch, trg_batch) in enumerate(train_loader):
            src_tensor = torch.LongTensor(src_batch).to(device)
            trg_tensor = torch.LongTensor(trg_batch).to(device)
            outputs = model.forward(src=src_tensor, trg=trg_tensor, train=teacher_force)

            outputs = outputs[:, 1:].contiguous().view(-1, outputs.shape[-1]).to(device)
            trg_tensor = trg_tensor[:, 1:].contiguous().view(-1).to(device)
            loss = criterion(outputs, trg_tensor)

            average_loss.append(loss.item())

            loss.backward()
            if (k + 1) % 5 == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()

            if ((k + 1) % 10 == 0):
                print("Epoch: {:d} batch step: [{:d}/{:d}] Loss: {:.4f}".format(epoch + 1, k + 1, len(train_loader),
                                                                                np.mean(average_loss)))
        print("\nEpoch: {:d}  Average Loss: {:.4f} +- {:.4f}\n".format(epoch + 1, np.mean(average_loss),
                                                                       np.std(average_loss)))

        evaluate(test_loader=test_loader, criterion=criterion, model=model, device=device)
#         match_score = evaluate(test_loader=test_loader, criterion=criterion, model=model, device=device,
#                                match_score=match_score)
