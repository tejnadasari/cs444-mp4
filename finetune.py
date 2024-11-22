import numpy as np
from torch import nn
import torch
from vision_transformer import vit_b_32, ViT_B_32_Weights
from tqdm import tqdm
import numpy as np


def get_encoder(name):
    if name == 'vit_b_32':
        torch.hub.set_dir("model")
        model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    return model


class ViTLinear(nn.Module):
    def __init__(self, n_classes, encoder_name):
        super(ViTLinear, self).__init__()

        self.vit_b = [get_encoder(encoder_name)]

        # Reinitialize the head with a new layer
        self.vit_b[0].heads[0] = nn.Identity()
        self.linear = nn.Linear(768, n_classes)

    def to(self, device):
        super(ViTLinear, self).to(device)
        self.vit_b[0] = self.vit_b[0].to(device)

    def forward(self, x):
        with torch.no_grad():
            out = self.vit_b[0](x)
        y = self.linear(out)
        return y


class VPTDeep(nn.Module):
    def __init__(self, n_classes, encoder_name, num_prompts=10):
        super(VPTDeep, self).__init__()

        # Get pretrained ViT
        self.vit = get_encoder(encoder_name)

        # Freeze backbone parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Initialize prompts with proper initialization
        num_layers = len(self.vit.encoder.layers)
        hidden_dim = 768  # ViT-B hidden dimension
        prompt_dim = num_prompts * hidden_dim
        # Initialize using suggested formula v^2 = 6/(hidden_dim + prompt_dim)
        v = np.sqrt(6.0 / (hidden_dim + prompt_dim))

        # Create learnable prompts
        self.prompts = nn.Parameter(
            torch.empty(1, num_layers, num_prompts, hidden_dim).uniform_(-v, v)
        )

        # Replace classification head
        self.vit.heads = nn.Linear(hidden_dim, n_classes)

        # Initialize the new classification head
        nn.init.zeros_(self.vit.heads.weight)
        nn.init.zeros_(self.vit.heads.bias)

    def forward(self, x):
        # Process input and add CLS token
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Forward through encoder with prompts
        x = self.vit.encoder(x, self.prompts)

        # Take CLS token and classify
        x = x[:, 0]
        x = self.vit.heads(x)
        return x

def test(test_loader, model, device):
    model.eval()
    total_loss, correct, n = 0., 0., 0

    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        correct += (y_hat.argmax(dim=1) == y).float().mean().item()
        loss = nn.CrossEntropyLoss()(y_hat, y)
        total_loss += loss.item()
        n += 1
    accuracy = correct / n
    loss = total_loss / n
    return loss, accuracy


def inference(test_loader, model, device, result_path):
    """Generate predicted labels for the test set."""
    model.eval()

    predictions = []
    with torch.no_grad():
        for x, _ in tqdm(test_loader):
            x = x.to(device)
            y_hat = model(x)
            pred = y_hat.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

    with open(result_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {result_path}")


class Trainer():
    def __init__(self, model, train_loader, val_loader, writer,
                 optimizer, lr, wd, momentum,
                 scheduler, epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.writer = writer

        self.model.to(self.device)

        # Modified optimizer setup for VPT
        if isinstance(model, VPTDeep):
            # Only optimize prompts and classification head
            params = list(model.prompts.parameters()) + list(model.vit.heads.parameters())
            self.optimizer = torch.optim.SGD(params,
                                             lr=0.01,  # Suggested learning rate
                                             weight_decay=0.01,  # Suggested weight decay
                                             momentum=0.9)

            if scheduler == 'multi_step':
                # Learning rate drops at epochs 60 and 80
                self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=[60, 80], gamma=0.1)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(),
                                             lr=lr, weight_decay=wd,
                                             momentum=momentum)
            if scheduler == 'multi_step':
                self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=[60, 80], gamma=0.1)

    def train_epoch(self):
        self.model.train()
        total_loss, correct, n = 0., 0., 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = nn.CrossEntropyLoss()(y_hat, y)
            total_loss += loss.item()
            correct += (y_hat.argmax(dim=1) == y).float().mean().item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            n += 1
        return total_loss / n, correct / n

    def val_epoch(self):
        self.model.eval()
        total_loss, correct, n = 0., 0., 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            correct += (y_hat.argmax(dim=1) == y).float().mean().item()
            loss = nn.CrossEntropyLoss()(y_hat, y)
            total_loss += loss.item()
            n += 1
        accuracy = correct / n
        loss = total_loss / n
        return loss, accuracy

    def train(self, model_file_name, best_val_acc=-np.inf):
        best_epoch = np.NaN
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.val_epoch()
            self.writer.add_scalar('lr', self.lr_schedule.get_last_lr()[0], epoch)
            self.writer.add_scalar('val_acc', val_acc, epoch)
            self.writer.add_scalar('val_loss', val_loss, epoch)
            self.writer.add_scalar('train_acc', train_acc, epoch)
            self.writer.add_scalar('train_loss', train_loss, epoch)
            pbar.set_description(f"val acc: {val_acc:.4f}, train acc: {train_acc:.4f}", refresh=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_file_name)
            self.lr_schedule.step()

        return best_val_acc, best_epoch