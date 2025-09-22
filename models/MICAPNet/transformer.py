import torch
from torch import nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.rand(1, max_len, d_model))
        self.pe.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # x: (batch_size, seq_len, embedding_dim)
        return self.dropout(x)

class uni_Transformer(nn.Module):
    def __init__(self, input_size, num_classes,
                 d_model=256, n_head=8, n_layers_feat=1,
                 dropout=0.3, max_len=350):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.emb = nn.Linear(input_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_feat = nn.TransformerEncoder(layer, num_layers=n_layers_feat)

        self.fc_pred = nn.Linear(d_model, num_classes)

    def forward(self, x, seq_lengths):
        attn_mask = torch.stack([torch.cat([torch.zeros(len_, device=x.device),
                                 float('-inf')*torch.ones(max(seq_lengths)-len_, device=x.device)])
                                for len_ in seq_lengths])
        if x.shape[1] > self.max_len:
            attn_mask = attn_mask[:, :self.max_len]
            x = x[:, :self.max_len, :]

        x = self.emb(x) # * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        feat = self.model_feat(x, src_key_padding_mask=attn_mask)
        feat_ehr = feat

        padding_mask = torch.ones_like(attn_mask).unsqueeze(2)
        padding_mask[attn_mask==float('-inf')] = 0

        feat = (padding_mask * feat).sum(dim=1) / padding_mask.sum(dim=1)

        pred = self.fc_pred(feat).sigmoid()

        return feat_ehr, pred


class EHR_Transformer(nn.Module):
    def __init__(self, input_size,
                 d_model=256, n_head=8, n_layers_feat=1,
                 n_layers_shared=1, n_layers_distinct=1,
                 dropout=0.3, max_len=350):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.emb = nn.Linear(input_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_feat = nn.TransformerEncoder(layer, num_layers=n_layers_feat)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_shared = nn.TransformerEncoder(layer, num_layers=n_layers_shared)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_distinct = nn.TransformerEncoder(layer, num_layers=n_layers_distinct)


    def forward(self, x, seq_lengths):
        attn_mask = torch.stack([torch.cat([torch.zeros(len_, device=x.device),
                                 float('-inf')*torch.ones(max(seq_lengths)-len_, device=x.device)])
                                for len_ in seq_lengths])

        if x.shape[1] > self.max_len:
            attn_mask = attn_mask[:, :self.max_len]
            x = x[:, :self.max_len, :]

        x = self.emb(x) # * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        feat = self.model_feat(x, src_key_padding_mask=attn_mask)

        h_shared = self.model_shared(feat, src_key_padding_mask=attn_mask)
        h_distinct = self.model_distinct(feat, src_key_padding_mask=attn_mask)

        padding_mask = torch.ones_like(attn_mask).unsqueeze(2)
        padding_mask[attn_mask==float('-inf')] = 0

        feat_ehr = padding_mask * feat
        IB_feat_ehr = feat_ehr.mean(1)
        shared_feat = padding_mask * h_shared
        spe_feat = padding_mask * h_distinct

        shared_fc = (shared_feat).sum(dim=1) / padding_mask.sum(dim=1)
        spe_fc = (spe_feat).sum(dim=1) / padding_mask.sum(dim=1)

        return IB_feat_ehr, shared_feat, shared_fc, spe_feat, spe_fc


# ehr_model = EHR_Transformer(input_size=76, num_classes=25,
#                            d_model=256, n_head=4,
#                            n_layers_feat=1, n_layers_shared=1,
#                            n_layers_distinct=1,
#                            dropout=0.3)

# import numpy as np
# x = torch.rand((2, 20, 76))
# length = np.array([15, 20])
# length = torch.from_numpy(length)
# feat_ehr, shared_feat, shared_fc, spe_feat, spe_fc = ehr_model(x, length)
# print(feat_ehr.shape)
# print(shared_feat.shape)
# print(shared_fc.shape)
# print(spe_feat.shape)
# print(spe_fc.shape)

