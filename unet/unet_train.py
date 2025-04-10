import torch.nn as nn
import torch as torch
import torch.optim as optim
from tqdm import tqdm

########################## Loss Functions ##########################
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# Combined loss
bce_loss = nn.BCELoss()
dice_loss = DiceLoss()

def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)

############################ Training Loops ##########################
def train_model(model, train_loader, device, epochs=20, lr=1e-4, save_path="unet_binarization.pth"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)

            # Forward pass
            preds = model(imgs)
            loss = combined_loss(preds, masks)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

        # Optionally save after each epoch
        torch.save(model.state_dict(), save_path)
    return model