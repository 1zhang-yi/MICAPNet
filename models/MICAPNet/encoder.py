import torch
import torch.nn as nn

from models.MICAPNet.resnet import CXR_Resnet
from models.MICAPNet.transformer import EHR_Transformer

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        hidden_size = args.hidden_size
        self.ehr_model = EHR_Transformer(input_size=args.ehr_input_size,
                                        d_model=hidden_size, n_head=args.ehr_n_head,
                                        n_layers_feat=args.ehr_n_layers, n_layers_shared=args.ehr_n_layers,
                                        n_layers_distinct=args.ehr_n_layers,
                                        dropout=args.ehr_dropout)

        self.cxr_model = CXR_Resnet(hidden_size=hidden_size)

        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.fusion_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # self.shared_pred = nn.Linear(hidden_size, 25)
        # self.spe_ehr_pred = nn.Linear(hidden_size, 25)
        # self.spe_cxr_pred = nn.Linear(hidden_size, 25)


    def forward(self, x, img, seq_lengths):
        feat_ehr, shared_feat_ehr, shared_fc_ehr, spe_feat_ehr, spe_fc_ehr = self.ehr_model(x, seq_lengths)
        feat_cxr, shared_feat_cxr, shared_fc_cxr, spe_feat_cxr, spe_fc_cxr = self.cxr_model(img)

        shared_feat_ehr = self.shared_project(shared_feat_ehr)
        shared_feat_cxr = self.shared_project(shared_feat_cxr)
        fusion_shared_feat = torch.cat([shared_feat_ehr, shared_feat_cxr], dim=1)
        fusion_shared_feat = self.fusion_project(fusion_shared_feat)

        # ehr_pred = self.spe_ehr_pred(spe_fc_ehr)
        # cxr_pred = self.spe_cxr_pred(spe_fc_cxr)
        # shared_pred = self.shared_pred(fusion_shared_feat.sum(1))

        ehr_dict = {'IB_feat_ehr': feat_ehr,
                    'shared_feat_ehr': shared_feat_ehr,
                    'shared_fc_ehr': shared_fc_ehr,
                    'spe_feat_ehr': spe_feat_ehr,
                    'spe_fc_ehr': spe_fc_ehr
        }
        cxr_dict = {'IB_feat_cxr': feat_cxr,
                    'shared_feat_cxr': shared_feat_cxr,
                    'shared_fc_cxr': shared_fc_cxr,
                    'spe_feat_cxr': spe_feat_cxr,
                    'spe_fc_cxr': spe_fc_cxr
        }

        return fusion_shared_feat, ehr_dict, cxr_dict
        # return fusion_shared_feat, ehr_dict, cxr_dict, ehr_pred, cxr_pred, shared_pred

# if __name__ == '__main__':
#     import argparse
#     import numpy as np
#     parser = argparse.ArgumentParser(description='arguments')
#     parser.add_argument('--modality', default='mm', type=str, choices=['ehr', 'ehr_paired', 'mm'])
#     parser.add_argument('--model_name', type=str, default='DrFuse', help='lstm or transformer')
#
#     ################# ehr ####################
#     parser.add_argument('--ehr_data_dir', default='data/ehr', type=str, help='ehr data path')
#     parser.add_argument('--task', type=str, default='phenotyping',
#                         help='train or eval for in-hospital-mortality or phenotyping, decompensation, length-of-stay')
#     parser.add_argument('--timestep', type=float, default=1.0, help="fixed timestep used in the dataset")
#     parser.add_argument('--normalizer_state', type=str, default=None,
#                         help='Path to a state file of a normalizer. Leave none if you want to '
#                              'use one of the provided ones.')
#     parser.add_argument('--num_classes', type=int, default=25)
#     parser.add_argument('--get_header_csvpath', type=str, help='Directory of csv path',
#                         default='14991576_episode3_timeseries.csv')
#
#
#     ################# image ####################
#     parser.add_argument('--image_dir', default='data/MIMIC-CXR-JPG', type=str, help='ehr data path')
#
#     ################# ehr transformer model ####################
#     parser.add_argument('--ehr_input_size', type=int, default=76, help='number of hidden units')
#     parser.add_argument('--ehr_n_layers', type=int, default=2)
#     parser.add_argument('--ehr_n_head', type=int, default=4)
#     parser.add_argument('--hidden_size', type=int, default=256)
#     parser.add_argument('--ehr_dropout', type=float, default=0.1)
#
#     ################# train ####################
#     parser.add_argument('--epochs', type=int, default=100, help='number of chunks to train')
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
#     parser.add_argument('--wd', type=float, default=0)
#
#     ################# train ####################
#     parser.add_argument('--save_checkpoints', type=str, help='Directory relative which all output files are stored',
#                     default='checkpoints')
#
#     args = parser.parse_args()
#
#     model = Encoder(args)
#     img = torch.rand((2, 3, 224, 224))
#     x = torch.rand((2, 20, 76))
#     length = np.array([13, 20])
#     length = torch.from_numpy(length)
#     fusion, ehr_dict, cxr_dict = model(x, img, length)
#     print('.......................fusion........................')
#     print(fusion.shape)
#     print('.......................EHR........................')
#     print(ehr_dict['IB_feat_ehr'].shape)
#     print(ehr_dict['shared_feat_ehr'].shape)
#     print(ehr_dict['shared_fc_ehr'].shape)
#     print(ehr_dict['spe_feat_ehr'].shape)
#     print(ehr_dict['spe_fc_ehr'].shape)
#     print('.......................CXR........................')
#     print(cxr_dict['IB_feat_cxr'].shape)
#     print(cxr_dict['shared_feat_cxr'].shape)
#     print(cxr_dict['shared_fc_cxr'].shape)
#     print(cxr_dict['spe_feat_cxr'].shape)
#     print(cxr_dict['spe_fc_cxr'].shape)