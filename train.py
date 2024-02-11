import torch
import torch.nn as nn
from model import ThreeBodiesModel
from dataset import ThreeBodiesDataset

from tqdm import tqdm

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = ThreeBodiesDataset(device=device)
    train_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=100)
    model = ThreeBodiesModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(20)):

        train_loss, test_loss = 0, 0

        for X, in train_dataloader:
            
            X = X.to(device)
            X_pred = model(X)
            loss = loss_fn(X_pred, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
        train_loss /= len(train_dataloader)

        for X, in test_dataloader:

            X = X.to(device)
            with torch.inference_mode():
            
                X_pred = model(X)
                loss = loss_fn(X_pred, X)
            
            test_loss += loss
        test_loss /= len(test_dataloader)

        print(f'Train loss: {train_loss}, Test loss: {test_loss}')

    torch.save(model.state_dict(), 'models/model.pt')
        