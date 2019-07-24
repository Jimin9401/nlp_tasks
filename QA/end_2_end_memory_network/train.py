from eval import evaluate

import torch
import numpy as np
torch.manual_seed(777)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def train(model, iterator, optimizer, criterion, train_loader, test_loader=None):
    model = model.train()
    match_score = 0

    for epoch in range(iterator):
        average_loss = []
        for k, batch in enumerate(train_loader):
            memory_batch=batch.story
            query_batch=batch.query
            answer_batch=batch.answer

            memory_tensor = torch.LongTensor(memory_batch).to(device)
            query_tensor = torch.LongTensor(query_batch).to(device)
            optimizer.zero_grad()
            outputs = model.forward(memory=memory_tensor, query=query_tensor)

            loss = criterion(outputs.view(-1,outputs.size(-1)), answer_batch.view(-1))

            average_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if ((k + 1) % 2 == 0):
                print("Epoch: {:d} batch step: [{:d}/{:d}] Loss: {:.4f}".format(epoch + 1, k + 1, len(train_loader),
                                                                                np.mean(average_loss)))
        print("\nEpoch: {:d}  Average Loss: {:.4f} +- {:.4f}\n".format(epoch + 1, np.mean(average_loss),
                                                                       np.std(average_loss)))
        if test_loader:
            match_score = evaluate(test_loader=test_loader, criterion=criterion, model=model, device=device,
                                   match_score=match_score)
