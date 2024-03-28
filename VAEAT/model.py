import time

import torch
import torch.nn as nn

from utils import *

device = get_default_device()


class Encoder(nn.Module):
    def __init__(self, in_size, lstm_h_dim, latent_size):
        super().__init__()
        self.lstm_h_dim = lstm_h_dim
        self.lstm = nn.LSTM(in_size, lstm_h_dim, batch_first=True)
        self.mu = nn.Linear(lstm_h_dim, latent_size)
        self.log_var = nn.Linear(lstm_h_dim, latent_size)

    def reparameter(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, w):
        out, _ = self.lstm(w)
        w_omiga = torch.randn(w.size(0), self.lstm_h_dim, 1).to(device)
        H = torch.nn.Tanh()(out).to(device)
        weights = torch.nn.Softmax(dim=-1)(torch.bmm(H, w_omiga).squeeze()).unsqueeze(dim=-1).repeat(1, 1,
                                                                                                     self.lstm_h_dim)

        out = torch.mul(out, weights)

        output = out.reshape(w.size(0) * w.size(1), -1)
        mu = self.mu(output)
        log_var = self.log_var(output)
        z = self.reparameter(mu, log_var)
        z = z.view(w.size(0), w.size(1), -1)

        # kl divergence
        kl_loss = 0.5 * (1 + 2 * log_var - mu.pow(2) - torch.exp(2 * log_var))
        kl_loss = torch.sum(kl_loss)

        return z, kl_loss


class Decoder(nn.Module):
    def __init__(self, latent_size, lstm_h_dim, out_size):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

        self.lstm = nn.LSTM(latent_size, lstm_h_dim, batch_first=True)
        self.mu = nn.Linear(lstm_h_dim, out_size)
        self.log_var = nn.Linear(lstm_h_dim, out_size)

    def forward(self, z):
        batch_size = z.shape[0]
        window_size = z.shape[1]

        output, _ = self.lstm(z)
        output = output.reshape(z.size(0) * z.size(1), -1)
        mu = self.mu(output)
        log_var = self.log_var(output)
        w = self.sigmoid(log_var)
        return w.view(batch_size, window_size, -1)


class VAEATModel(nn.Module):
    def __init__(self, w_size, lstm_h_dim, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, lstm_h_dim, z_size)
        self.decoder1 = Decoder(z_size, lstm_h_dim, w_size)
        self.decoder2 = Decoder(z_size, lstm_h_dim, w_size)

    def training_step(self, batch, n):
        z, kl = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1)[0])
        loss1 = 1 / n * (torch.mean((batch - w1) ** 2) - 1.0 * kl) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * (torch.mean((batch - w2) ** 2) - 1.0 * kl) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    def validation_step(self, batch, n):
        z, kl = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1)[0])

        loss1 = 1 / n * (torch.mean((batch - w1) ** 2) - 1.0 * kl) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * (torch.mean((batch - w2) ** 2) - 1.0 * kl) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return {'val_loss1': loss1, 'val_loss2': loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))


def evaluate(model, val_loader, n):
    outputs = []
    for data in val_loader:
        data = data.to(torch.float32)
        data = to_device(data, device)
        # {val_loss1, val_loss2}
        outputs.append(model.validation_step(data, n))

    return model.validation_epoch_end(outputs)


def training(opt, epoch, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters()) + list(model.decoder2.parameters()))

    start_time = time.time()

    for data in train_loader:
        # data = torch.stack(data, dim=1)
        data = data.to(torch.float32)
        data = to_device(data, device)

        # Train VAE1
        loss1, loss2 = model.training_step(data, epoch + 1)
        loss1.backward()
        optimizer1.step()
        optimizer1.zero_grad()

        # Train VAE2
        loss1, loss2 = model.training_step(data, epoch + 1)
        loss2.backward()
        optimizer2.step()
        optimizer2.zero_grad()

    # {loss1, loss2}
    result = evaluate(model, val_loader, epoch + 1)
    model.epoch_end(epoch, result)

    history.append(result)

    total_train_time = time.time() - start_time

    return total_train_time, history


def testing(model, test_loader, alpha=.5, beta=.5):
    results = []
    pred_time = []
    for batch in test_loader:
        start_time = time.time()

        batch = batch.to(torch.float32)
        batch = to_device(batch, device)

        z, kl = model.encoder(batch)
        w1 = model.decoder1(z)
        w2 = model.decoder2(z)
        results.append(alpha * torch.mean(torch.mean((batch - w1) ** 2, axis=2), axis=1) + beta * torch.mean(
            torch.mean((batch - w2) ** 2, axis=2), axis=1))

        pred_time.append(time.time() - start_time)
    return results, np.mean(pred_time)
