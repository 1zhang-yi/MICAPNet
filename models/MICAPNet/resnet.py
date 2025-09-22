import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50 as Resnet

class uni_resnet(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()

        resnet_ = Resnet()
        self.cxr_model_feat = nn.Sequential(
            resnet_.conv1,
            resnet_.bn1,
            resnet_.relu,
            resnet_.maxpool,
        )
        resnet_ = Resnet()
        self.cxr_model_pred = nn.Sequential(
            resnet_.layer1,
            resnet_.layer2,
            resnet_.layer3,
            resnet_.layer4,
            resnet_.avgpool,
            nn.Flatten(),
        )
        self.cxr_model_pred.fc = nn.Sequential(
            nn.Linear(in_features=resnet_.fc.in_features, out_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=num_classes)
        )

    def forward(self, img):

        feat_cxr = self.cxr_model_feat(img)
        pred = self.cxr_model_pred(feat_cxr)

        return pred


class CXR_Resnet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        resnet_ = Resnet()
        self.shallow_feat = nn.Sequential(
            resnet_.conv1,
            resnet_.bn1,
            resnet_.relu,
            resnet_.maxpool,
            resnet_.layer1,
        )

        ###################### shared branch ###################
        resnet_ = Resnet()
        self.cxr_shared_feat = nn.Sequential(
            resnet_.layer2,
            resnet_.layer3,
            resnet_.layer4,
        )
        resnet_ = Resnet()
        self.cxr_shared_fc = nn.Sequential(
            resnet_.avgpool,
            nn.Flatten(),
            nn.Linear(in_features=resnet_.fc.in_features, out_features=hidden_size),
        )

        ###################### specific branch ###################
        resnet_ = Resnet()
        self.cxr_spe_feat = nn.Sequential(
            resnet_.layer2,
            resnet_.layer3,
            resnet_.layer4,
        )
        resnet_ = Resnet()
        self.cxr_spe_fc = nn.Sequential(
            resnet_.avgpool,
            nn.Flatten(),
            nn.Linear(in_features=resnet_.fc.in_features, out_features=hidden_size),
        )

        self.shared_feat = nn.Sequential(
            nn.Conv2d(in_channels=resnet_.fc.in_features, out_channels=resnet_.fc.in_features//2, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(resnet_.fc.in_features//2),
            nn.Conv2d(in_channels=resnet_.fc.in_features//2, out_channels=resnet_.fc.in_features // 2, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(resnet_.fc.in_features // 2),
            nn.Conv2d(in_channels=resnet_.fc.in_features//2, out_channels=hidden_size, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(hidden_size),
        )

        self.spe_feat = nn.Sequential(
            nn.Conv2d(in_channels=resnet_.fc.in_features, out_channels=resnet_.fc.in_features // 2, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(resnet_.fc.in_features // 2),
            nn.Conv2d(in_channels=resnet_.fc.in_features // 2, out_channels=resnet_.fc.in_features // 2, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(resnet_.fc.in_features // 2),
            nn.Conv2d(in_channels=resnet_.fc.in_features // 2, out_channels=hidden_size, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(hidden_size),
        )

        self.feat_cxr = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, img):

        feat_cxr = self.shallow_feat(img)

        shared_feat = self.cxr_shared_feat(feat_cxr)
        shared_fc = self.cxr_shared_fc(shared_feat)

        spe_feat = self.cxr_spe_feat(feat_cxr)
        spe_fc = self.cxr_spe_fc(spe_feat)

        shared_feat = self.shared_feat(shared_feat)
        spe_feat = self.spe_feat(spe_feat)
        shared_feat = torch.flatten(shared_feat, start_dim=2).transpose(1, 2)
        spe_feat = torch.flatten(spe_feat, start_dim=2).transpose(1, 2)

        feat_cxr = self.feat_cxr(feat_cxr)
        feat_cxr = self.avgpool(feat_cxr)
        feat_cxr = torch.flatten(feat_cxr, start_dim=1)

        return feat_cxr, shared_feat, shared_fc, spe_feat, spe_fc



# model = CXR_Resnet(256, 14)
# x = torch.rand((2, 3, 224, 224))
# feat_cxr, shared_feat, shared_fc, spe_feat, spe_fc = model(x)
# print(feat_cxr.shape)
# print(shared_feat.shape)
# print(shared_fc.shape)
# print(spe_feat.shape)
# print(spe_fc.shape)




