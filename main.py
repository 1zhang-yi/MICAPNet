import numpy as np
import argparse
from tqdm import tqdm
import os
import datetime
import torch
from itertools import chain
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR
import torch.nn as nn

from utils import createLogger
from metrics import computeAUROC
from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.mm_dataset_paired import get_data_loader, load_discretized_header

from models.MICAPNet.encoder import Encoder
from models.mi_estimators.CLUB import CLUBSample
from models.mi_estimators.InfoNCE import InfoNCE, approx_infoNCE_loss
from models.MICAPNet.SFCAmodule import CrossFusion
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def mi_first_forward(Text_mi_club, Image_mi_club, optimizer_text_mi_net,
                      optimizer_image_mi_net, ehr_shared, ehr_spe, cxr_shared, cxr_spe, mi_iters):
    for i in range(mi_iters):
        optimizer_text_mi_net.zero_grad()
        optimizer_image_mi_net.zero_grad()

        lld_text_loss = -Text_mi_club.loglikeli(ehr_shared, ehr_spe)
        lld_text_loss.backward()
        optimizer_text_mi_net.step()

        lld_image_loss = -Image_mi_club.loglikeli(cxr_shared, cxr_spe)
        lld_image_loss.backward()
        optimizer_image_mi_net.step()

    return optimizer_text_mi_net, optimizer_image_mi_net


def mi_second_forward(shared_fusion_feat, ehr_dict, cxr_dict, Fusion, Text_mi_club, Image_mi_club, Text_IB_club, Image_IB_club,
                      pred_criterion, optimizer, optimizer_text_IB_net, optimizer_image_IB_net, label):
    ehr_shared_fc = ehr_dict['shared_fc_ehr']
    ehr_spe_fc = ehr_dict['spe_fc_ehr']
    cxr_shared_fc = cxr_dict['shared_fc_cxr']
    cxr_spe_fc = cxr_dict['spe_fc_cxr']

    # concat_feat = torch.cat([ehr_shared_fc, cxr_shared_fc], dim=0)
    # nce_loss = InfoNCE_loss(concat_feat)
    ########################## disentanglement loss ###################################
    softmax = nn.Softmax()
    nce_loss = approx_infoNCE_loss(softmax(ehr_shared_fc), softmax(cxr_shared_fc))
    dis_text_mi = Text_mi_club(softmax(ehr_shared_fc), softmax(ehr_spe_fc))
    dis_image_mi = Image_mi_club(softmax(cxr_shared_fc), softmax(cxr_spe_fc))
    disentangled_loss = 0.4 * (dis_text_mi + dis_image_mi) + 0.6 * nce_loss

    IB_input_ehr = ehr_dict['IB_feat_ehr']
    IB_input_cxr = cxr_dict['IB_feat_cxr']
    ehr_spe_feat = ehr_dict['spe_feat_ehr']
    cxr_spe_feat = cxr_dict['spe_feat_cxr']

    pred, ehr_pred, cxr_pred, fusion_feat, IB_ehr_feat, IB_cxr_feat = Fusion(shared_fusion_feat, ehr_spe_feat, cxr_spe_feat)

    ############################ multimodal IB loss ###################################
    optimizer_text_IB_net.zero_grad()
    optimizer_image_IB_net.zero_grad()

    lld_text_loss = -Text_IB_club.loglikeli(IB_input_ehr.detach(), IB_ehr_feat.detach())
    lld_text_loss.backward()
    optimizer_text_IB_net.step()

    lld_image_loss = -Image_IB_club.loglikeli(IB_input_cxr.detach(), IB_cxr_feat.detach())
    lld_image_loss.backward()
    optimizer_image_IB_net.step()

    compress_text_mi = Text_IB_club(softmax(IB_input_ehr), softmax(IB_ehr_feat))
    compress_image_mi = Image_IB_club(softmax(IB_input_cxr), softmax(IB_cxr_feat))
    pred_ehr_loss = pred_criterion(ehr_pred, label)
    pred_cxr_loss = pred_criterion(cxr_pred, label)
    IB_loss = (pred_ehr_loss + pred_cxr_loss) + 0.1 * (compress_text_mi + compress_image_mi)

    ############################ task loss #####################################
    pred_loss = pred_criterion(pred, label)

    loss = 0.3 * disentangled_loss + pred_loss + 0.2 * IB_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, pred



def train_epoch(args, Text_Image_encoder, Fusion, Text_mi_club, Image_mi_club, Text_IB_club, Image_IB_club, train_loader,
                pred_criterion, InfoNCE_loss, optimizer, optimizer_text_mi_net, optimizer_image_mi_net,
                optimizer_text_IB_net, optimizer_image_IB_net, epoch):
    Text_Image_encoder.train()
    Fusion.train()
    Text_mi_club.train()
    Image_mi_club.train()
    Text_IB_club.train()
    Image_IB_club.train()

    epoch_loss = 0.0

    train_loader = tqdm(train_loader)
    for i, data in enumerate(train_loader):
        x = data[0].float().cuda()
        image = data[1].float().cuda()
        label = data[2].float().cuda()
        seq_length = data[3]

        fusion_feat, ehr_dict, cxr_dict = Text_Image_encoder(x, image, seq_length)
        ehr_shared_fc = ehr_dict['shared_fc_ehr']
        ehr_spe_fc = ehr_dict['spe_fc_ehr']
        cxr_shared_fc = cxr_dict['shared_fc_cxr']
        cxr_spe_fc = cxr_dict['spe_fc_cxr']


        optimizer_text_mi_net, optimizer_image_mi_net = \
        mi_first_forward(Text_mi_club, Image_mi_club,
                         optimizer_text_mi_net, optimizer_image_mi_net,
                         ehr_shared_fc.detach(), ehr_spe_fc.detach(), cxr_shared_fc.detach(), cxr_spe_fc.detach(), args.mi_iters)


        batch_loss, batch_pred = mi_second_forward(fusion_feat, ehr_dict, cxr_dict, Fusion, Text_mi_club, Image_mi_club, Text_IB_club, Image_IB_club,
                                                   pred_criterion, optimizer, optimizer_text_IB_net, optimizer_image_IB_net, label)
        epoch_loss += batch_loss.item()
        train_loader.set_postfix(train_loss=batch_loss.item())

    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss

def test_epoch(Text_Image_encoder, Fusion, Text_mi_club, Image_mi_club, Text_IB_club, Image_IB_club, test_loader):
    Text_Image_encoder.eval()
    Fusion.eval()
    Text_mi_club.eval()
    Image_mi_club.eval()
    Text_IB_club.eval()
    Image_IB_club.eval()

    outlabel = torch.FloatTensor().cuda()
    outpred = torch.FloatTensor().cuda()

    test_loader = tqdm(test_loader)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data[0].float().cuda()
            image = data[1].float().cuda()
            label = data[2].float().cuda()
            seq_length = data[3]

            shared_fusion_feat, ehr_dict, cxr_dict = Text_Image_encoder(x, image, seq_length)
            ehr_spe_feat = ehr_dict['spe_feat_ehr']
            cxr_spe_feat = cxr_dict['spe_feat_cxr']

            pred, ehr_pred, cxr_pred, fusion_feat, IB_ehr_feat, IB_cxr_feat = Fusion(shared_fusion_feat, ehr_spe_feat, cxr_spe_feat)

            outpred = torch.cat((outpred, pred), 0)
            outlabel = torch.cat((outlabel, label), 0)

    test_metrics = computeAUROC(outlabel.data.cpu().numpy(), outpred.data.cpu().numpy())
    return test_metrics

def save_models(Text_Image_encoder, Fusion, Text_mi_club, Image_mi_club,
                optimizer, optimizer_text_mi_net, optimizer_image_mi_net, epoch, path):
    state_dict = {
        'Text_Image_encoder': Text_Image_encoder.state_dict(),
        'Fusion': Fusion.state_dict(),
        'Text_mi_club': Text_mi_club.state_dict(),
        'Image_mi_club': Image_mi_club.state_dict(),
        'Text_IB_club': Text_IB_club.state_dict(),
        'Image_IB_club': Image_IB_club.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_text_mi_net': optimizer_text_mi_net.state_dict(),
        'optimizer_image_mi_net': optimizer_image_mi_net.state_dict(),
        'optimizer_text_IB_net': optimizer_text_IB_net.state_dict(),
        'optimizer_image_IB_net': optimizer_image_IB_net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state_dict, path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--modality', default='mm', type=str, choices=['ehr', 'ehr_paired', 'mm'])
    parser.add_argument('--model_name', type=str, default='MICAPNet')

    ################# ehr ####################
    parser.add_argument('--ehr_data_dir', default='data/ehr', type=str, help='ehr data path')
    parser.add_argument('--task', type=str, default='phenotyping',
                        help='train or eval for in-hospital-mortality or phenotyping, decompensation, length-of-stay')
    parser.add_argument('--timestep', type=float, default=1.0, help="fixed timestep used in the dataset")
    parser.add_argument('--normalizer_state', type=str, default=None,
                        help='Path to a state file of a normalizer. Leave none if you want to '
                             'use one of the provided ones.')
    parser.add_argument('--ehr_num_classes', type=int, default=25)
    parser.add_argument('--get_header_csvpath', type=str, help='Directory of csv path',
                        default='14991576_episode3_timeseries.csv')

    ################# image ####################
    parser.add_argument('--image_dir', default='data/MIMIC-CXR-JPG', type=str, help='ehr data path')

    ################# ehr transformer model ####################
    parser.add_argument('--ehr_input_size', type=int, default=76, help='number of hidden units')
    parser.add_argument('--ehr_n_layers', type=int, default=2)
    parser.add_argument('--ehr_n_head', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--ehr_dropout', type=float, default=0.1)

    ################# SFCA fusion module ####################
    parser.add_argument('--hidden_sz', default=256, type=int)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--intermediate_size', default=256, type=int)
    parser.add_argument('--num_classes', default=25, type=int)
    parser.add_argument('--dropout', type=float, default=0.1)

    ################# train ####################
    parser.add_argument('--epochs', type=int, default=50, help='number of chunks to train')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.07, help='infonce temperature')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--mi_lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--mi_iters', type=int, default=2)
    parser.add_argument('--model_resume', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)

    ################# train ####################
    parser.add_argument('--save_checkpoints', type=str, help='Directory relative which all output files are stored',
                        default='checkpoints')

    args = parser.parse_args()

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    args.save_dir = f'{args.save_checkpoints}/{args.modality}/{args.model_name}_{formatted_datetime}'
    os.makedirs(args.save_dir, exist_ok=True)
    args.train_logs = f'{args.save_dir}/train_logs.txt'

    args.train_mm_pkl_fpath_paired = f'{args.ehr_data_dir}/{args.task}/mm_paired_train.pkl'
    args.test_mm_pkl_fpath_paired = f'{args.ehr_data_dir}/{args.task}/mm_paired_test.pkl'

    logger = createLogger(args)
    logger.write(f"The saved training checkpoints and logs will be save at {args.save_dir}.")

    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)

    ############## prepare EHR datasetloader ####################
    discretizer = Discretizer(timestep=float(args.timestep),
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')
    discretizer_header = load_discretized_header(discretizer, args.get_header_csvpath)
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'normalizers/ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(args.timestep)
    normalizer.load_params(normalizer_state)

    train_loader, test_loader = get_data_loader(discretizer, normalizer, args)

    ############## prepare models ####################
    Text_Image_encoder = Encoder(args).cuda()
    Fusion = CrossFusion(args).cuda()

    Text_mi_club = CLUBSample(args.hidden_size, args.hidden_size, args.hidden_size).cuda()
    Image_mi_club = CLUBSample(args.hidden_size, args.hidden_size, args.hidden_size).cuda()

    Text_IB_club = CLUBSample(args.hidden_size, args.hidden_size, args.hidden_size).cuda()
    Image_IB_club = CLUBSample(args.hidden_size, args.hidden_size, args.hidden_size).cuda()

    ############## prepare optimizers ####################
    optimizer = torch.optim.Adam(chain(Text_Image_encoder.parameters(), Fusion.parameters()),
                                 lr=args.lr, weight_decay=args.wd)
    optimizer_text_mi_net = torch.optim.Adam(Text_mi_club.parameters(), lr=args.mi_lr)
    optimizer_image_mi_net = torch.optim.Adam(Image_mi_club.parameters(), lr=args.mi_lr)

    optimizer_text_IB_net = torch.optim.Adam(Text_IB_club.parameters(), lr=args.mi_lr)
    optimizer_image_IB_net = torch.optim.Adam(Image_IB_club.parameters(), lr=args.mi_lr)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    ############## prepare loss function ####################
    pred_criterion = nn.BCELoss()
    InfoNCE_loss = InfoNCE(temperature=args.temperature)

    ############### resume ##################################
    start_epoch = args.start_epoch
    if args.model_resume is not None:
        checkpoints = torch.load(args.model_resume)
        Text_Image_encoder.load_state_dict(checkpoints['Text_Image_encoder'])
        Fusion.load_state_dict(checkpoints['Fusion'])
        Text_mi_club.load_state_dict(checkpoints['Text_mi_club'])
        Image_mi_club.load_state_dict(checkpoints['Image_mi_club'])

        Text_IB_club.load_state_dict(checkpoints['Text_IB_club'])
        Image_IB_club.load_state_dict(checkpoints['Image_IB_club'])

        optimizer.load_state_dict(checkpoints['optimizer'])
        optimizer_text_mi_net.load_state_dict(checkpoints['optimizer_text_mi_net'])
        optimizer_image_mi_net.load_state_dict(checkpoints['optimizer_image_mi_net'])

        optimizer_text_IB_net.load_state_dict(checkpoints['optimizer_text_IB_net'])
        optimizer_image_IB_net.load_state_dict(checkpoints['optimizer_image_IB_net'])

        start_epoch = checkpoints['epoch']
        logger.write("Resume from number {}-th model.".format(start_epoch))

    ########################## Start training #######################
    '''Training and Evaluation'''
    best_auroc = 0.0
    best_auprc = 0.0
    for epoch in range(start_epoch + 1, args.epochs):
        epoch_loss = train_epoch(args, Text_Image_encoder, Fusion, Text_mi_club, Image_mi_club, Text_IB_club, Image_IB_club, train_loader,
                                 pred_criterion, InfoNCE_loss,
                                 optimizer, optimizer_text_mi_net, optimizer_image_mi_net, optimizer_text_IB_net, optimizer_image_IB_net, epoch)
        scheduler.step()
        logger.write(f'Epoch {epoch}: {epoch_loss}.')

        test_metrics = test_epoch(Text_Image_encoder, Fusion, Text_mi_club, Image_mi_club, Text_IB_club, Image_IB_club, test_loader)

        auroc_scores = test_metrics['auroc_scores']
        auprc_scores = test_metrics['auprc_scores']
        auroc_mean = test_metrics['auroc_mean']
        auprc_mean = test_metrics['auprc_mean']
        ci_auroc = test_metrics['ci_auroc']
        ci_auprc = test_metrics['ci_auprc']

        if auroc_mean > best_auroc or auprc_mean > best_auprc:
            best_auroc = auroc_mean
            best_auprc = auprc_mean
            logger.write(f'Epoch {epoch}: test metrices:')
            logger.write(f'auroc_mean: {auroc_mean}')
            logger.write(f'auprc_mean: {auprc_mean}')

            logger.write(f'auroc_scores per label:')
            for i in range(len(auroc_scores)):
                logger.write(str(auroc_scores[i]))

            logger.write(f'auroc_scores per label lower and upper:')
            for i in range(len(ci_auroc)):
                logger.write(f'({ci_auroc[i][0]}, {ci_auroc[i][1]})')

            logger.write(f'auprc_scores per label:')
            for i in range(len(auprc_scores)):
                logger.write(str(auprc_scores[i]))

            logger.write(f'auprc_scores per label lower and upper:')
            for i in range(len(ci_auprc)):
                logger.write(f'({ci_auprc[i][0]}, {ci_auprc[i][1]})')

            auroc = "{:.5f}".format(auroc_mean)
            auprc = "{:.5f}".format(auprc_mean)

            save_name = f'epoch{epoch}_auroc{auroc}_auprc{auprc}.pth'
            save_path = f'{args.save_dir}/{save_name}'
            save_models(Text_Image_encoder, Fusion, Text_mi_club, Image_mi_club,
                        optimizer, optimizer_text_mi_net, optimizer_image_mi_net, epoch, save_path)
            logger.write(f'Finding the better modal, save in {save_path}')
