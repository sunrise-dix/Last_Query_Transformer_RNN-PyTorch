import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import roc_auc_score
from last_query_model import last_query_model


class ModelTrainer(pl.LightningModule):
    def __init__(self, dim_model, heads_en, total_ex, total_cat, total_in, seq_len, use_lstm=True):
        super().__init__()
        self.model = last_query_model(
            dim_model, heads_en, total_ex, total_cat, total_in, seq_len, use_lstm)

    def forward(self, in_ex, in_cat, in_in, labels):
        return self.model(in_ex, in_cat, in_in, labels)

    def training_step(self, batch, batch_idx):
        in_ex, in_cat, in_in, labels = batch
        predictions, _ = self(in_ex, in_cat, in_in, labels)
        labels = labels.float()
        # labels = labels.float()[:, -1].view(-1, 1)
        loss = F.mse_loss(predictions, labels)

        # labels_np = labels.cpu().detach(
        # ).numpy()
        # unique_labels = np.unique(labels_np)
        # if len(unique_labels) > 1:
        #     print(labels_np)
        #     auc_score = roc_auc_score(
        #         labels_np, predictions.cpu().detach().numpy())
        #     print('auc_score', auc_score)

        self.log('train_loss', loss)
        return {"loss": loss, "outs": predictions, "labels": labels}

    def validation_step(self, batch, batch_idx):
        in_ex, in_cat, in_in, labels = batch
        predictions, _ = self(in_ex, in_cat, in_in, labels)
        labels = labels.float()
        # labels = labels.float()[:, -1].view(-1, 1)
        loss = F.mse_loss(predictions, labels)

        labels_np = labels.cpu().detach(
        ).numpy()
        unique_labels = np.unique(labels_np)
        if len(unique_labels) > 1:
            auc_score = roc_auc_score(
                labels_np, predictions.cpu().detach().numpy())
            print(labels.shape, predictions.shape)
            print('auc_score', auc_score)

        self.log('val_loss', loss)
        return {"val_loss": loss, "outs": predictions, "labels": labels}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
