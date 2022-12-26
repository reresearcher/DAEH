import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.nn import functional as F

import timm

def guassianMatrix(X, sigma):
    G = torch.mm(X, X.t())
    K = 2 * G - torch.diag(G, 0).repeat(X.size(0), 1)
    K_final = torch.exp((1 / (2 * sigma * sigma)) * (K - torch.diag(G, 0).repeat(X.size(0), 1)))
    fa = torch.zeros(1).type(torch.ByteTensor).cuda()
    if fa in torch.isreal(K_final):
        K_final = torch.real(K_final)
    return K_final


def mutual_information(variable1, variable2, sigma1, sigma2, alpha):
    K_x = guassianMatrix(variable1, sigma1) / variable1.size(0)
    K_x = torch.where(torch.isnan(K_x), torch.full_like(K_x, 0), K_x)
    K_x = torch.where(torch.isinf(K_x), torch.full_like(K_x, 0), K_x)
    L_x, _ = torch.eig(K_x)
    lambda_x = torch.abs(L_x)
    H_x = (1 / (1 - alpha)) * torch.log((torch.sum(torch.pow(lambda_x, alpha))))

    K_y = guassianMatrix(variable2, sigma2) / variable2.size(0)
    K_y = torch.where(torch.isnan(K_y), torch.full_like(K_y, 0), K_y)
    K_y = torch.where(torch.isinf(K_y), torch.full_like(K_y, 0), K_y)
    L_y, _ = torch.eig(K_y)
    lambda_y = torch.abs(L_y)
    H_y = (1 / (1 - alpha)) * torch.log((torch.sum(torch.pow(lambda_y, alpha))))

    K_xy = K_x * K_y * variable1.size(0)
    K_xy = torch.where(torch.isnan(K_xy), torch.full_like(K_xy, 0), K_xy)
    K_xy = torch.where(torch.isinf(K_xy), torch.full_like(K_xy, 0), K_xy)
    L_xy, _ = torch.eig(K_xy)
    lambda_xy = torch.abs(L_xy)
    H_xy = (1 / (1 - alpha)) * torch.log((torch.sum(torch.pow(lambda_xy, alpha))))

    mutual_information = H_x + H_y - H_xy

    return mutual_information

def load_model(arch, code_length, txt_dim):
    """
    Load CNN model.

    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    
    if arch == 'alexnet':
        model0 = models.alexnet(pretrained=True)
        model = models.alexnet(pretrained=True)
        model0 = ModelWrapper0(model0)
        model.classifier = model.classifier[:-2]
        model = ModelWrapper(model, 4096, code_length)        
    elif arch == 'resnet152':
        model0 = models.resnet152(pretrained=True)
        model = models.resnet152(pretrained=True)
        model0 = ModelWrapper0(model0)
        model = nn.Sequential(*list(model.children())[:-1])
        model = ModelWrapper(model, 2048, code_length)

    elif arch == 'transformer':
        model0 = timm.create_model('vit_base_patch16_224', pretrained=True)
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.reset_classifier(0)
        model = ModelWrapper(model, 768, code_length)
    else:
        raise ValueError("Invalid model name!")

    return model, model0


class ModelWrapper0(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model):
        super(ModelWrapper0, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

class ModelWrapperFea(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length):
        super(ModelWrapperFea, self).__init__()
        self.model = model
        self.code_length = code_length
        self.extract_features = False

    def forward(self, x):
        return torch.squeeze(self.model(x))#, self.model.yuanclassifier(self.model(x))
    
    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag
        

class ModelWrapper(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.code_length = code_length

        self.hash_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )
        self.fake_attribute = nn.Sequential(
            nn.Linear(code_length, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1000),
            nn.ReLU(inplace=True),
        )
        self.attribute_code = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, code_length),
            nn.Tanh(),
        )
        # Extract features
        self.extract_features = False

    def forward(self, x):
        if self.extract_features:
            return torch.squeeze(self.model(x))#, self.model.yuanclassifier(self.model(x))
        else:
            return self.hash_layer(torch.squeeze(self.model(x))), \
                   self.fake_attribute(self.hash_layer(torch.squeeze(self.model(x)))), \
                   self.attribute_code(self.fake_attribute(self.hash_layer(torch.squeeze(self.model(x)))))
            # hash code, fake attribute, reconstruct code
    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag

class ModelWrapper_enc(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length):
        super(ModelWrapper_enc, self).__init__()
        self.model = model
        self.code_length = code_length

        self.ce_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )

    def forward(self, x):

        return self.ce_layer(torch.squeeze(self.model(x)))

class myImgNet(nn.Module):
    def __init__(self, code_len):
        super(myImgNet, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.view(x.size(0), -1).float()
        feat1 = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat1))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class myTxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(myTxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std=0.3)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class load_txt_encoder_pei(nn.Module):

    def __init__(self, dimIn, dimOut):
        self.inFea = dimIn
        self.code_length = dimOut

        super(load_txt_encoder_pei, self).__init__()

        self.fc1 = nn.Linear(dimIn, 4096)
        self.fc_encode = nn.Linear(4096, dimOut)
        self.dropout = nn.Dropout(p=0.5)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std= 0.3)

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = torch.tanh(hid)
        return code