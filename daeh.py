# encoding: utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger

from model_loader import load_model, myImgNet, myTxtNet, load_txt_encoder_pei, mutual_information
from evaluate import mean_average_precision

import sklearn.manifold as manifold_tools
from scipy.spatial.distance import pdist, squareform
import pickle as pkl


def tri_loss(code_I, code_T, code_Tea, S, I, ceshi_tri, ceshi_tea):

    B_I = F.normalize(code_I)
    B_T = F.normalize(code_T)
    B_Tea = F.normalize(code_Tea)

    BI_BI = B_I.mm(B_I.t())
    BT_BT = B_T.mm(B_T.t())
    BTea_BTea = B_Tea.mm(B_Tea.t())

    ###################################################
    BI_BT = B_I.mm(B_T.t())
    diagonal = BI_BT.diagonal()
    all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
    loss_pair0 = F.mse_loss(diagonal, 1.5 * all_1)

    BI_BTea = B_I.mm(B_Tea.t())
    diagonal = BI_BTea.diagonal()
    all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
    loss_pair1 = F.mse_loss(diagonal, 1.5 * all_1)

    BT_BTea = B_T.mm(B_Tea.t())
    diagonal = BT_BTea.diagonal()
    all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
    loss_pair2 = F.mse_loss(diagonal, 1.5 * all_1)

    loss_pair = ceshi_tri*loss_pair0 + ceshi_tea*(loss_pair1 + loss_pair2)
    ###################################################


    loss_dis =  ceshi_tri*F.mse_loss(BI_BT * (1 - I), S * (1 - I)) +\
                ceshi_tea*F.mse_loss(BI_BTea * (1 - I), S * (1 - I)) +\
                ceshi_tea*F.mse_loss(BT_BTea * (1 - I), S * (1 - I))
    
    loss_cons = ceshi_tri*F.mse_loss(BI_BI, BT_BT)+ \
               ceshi_tea*F.mse_loss(BI_BI, BTea_BTea)+ \
               ceshi_tea*F.mse_loss(BT_BT, BTea_BTea)


    loss = loss_pair + loss_dis + loss_cons

    return loss
def cross_loss(code_I, code_T, S, I):
    B_I = F.normalize(code_I)
    B_T = F.normalize(code_T)

    BI_BI = B_I.mm(B_I.t())
    BT_BT = B_T.mm(B_T.t())
    BI_BT = B_I.mm(B_T.t())
    # pdb.set_trace()
    diagonal = BI_BT.diagonal()
    all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
    loss_pair = F.mse_loss(diagonal, 1.5 * all_1)

    loss_dis_2 = F.mse_loss(BI_BT * (1 - I), S * (1 - I))

    loss_cons = F.mse_loss(BT_BT, BI_BI)

    loss = loss_pair + (loss_dis_2) + loss_cons

    return loss

def process(
          near_neighbor,
          num_train,
          batch_size,
          dataset,
          train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          multi_labels,
          code_length,
          feature_dim,
          label_dim,
          alpha,
          beta,
          gamma,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          evaluate_interval,
          snapshot_interval,
          topk,
          txt_dim,
          ceshi_tri,
          ceshi_cross,
          a1,
          ceshi_tea
          ):
          
    logger.info('xincan')
    """
    Training model.

    Args
        train_dataloader(torch.evaluate.data.DataLoader): Training data loader.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        multi_labels(bool): True, if dataset is multi-labels.
        code_length(int): Hash code length.
        feature_dim(int): Number of features.
        alpha, beta(float): Hyper-parameters.
        max_iter(int): Number of iterations.
        arch(str): Model name.
        lr(float): Learning rate.
        device(torch.device): GPU or CPU.
        verbose(bool): Print log.
        evaluate_interval(int): Interval of evaluation.
        snapshot_interval(int): Interval of snapshot.
        topk(int): Calculate top k data points map.
        checkpoint(str, optional): Paht of checkpoint.

    Returns
        None
    """

    # Model, optimizer, criterion
    gamma = 10 ** (gamma)
    logger.info('[gamma:{:.6f}]'.format(gamma))
    
    model, model0 = load_model(arch, code_length, txt_dim)
    img_encoder = myImgNet(code_length)
    txt_encoder = myTxtNet(code_length, txt_dim)

    model.to(device)
    model0.to(device)
    txt_encoder.to(device)
    img_encoder.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    optimizer_txt = optim.SGD(txt_encoder.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer_img = optim.SGD(img_encoder.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    resume_it = 0

    print('Extract features...')
    attribute_dim = 1000
    feature_dim = 4096
    features, texts, attributes, labels = extract_features(model, model0, train_dataloader, feature_dim, attribute_dim, device, verbose, label_dim,txt_dim)
    vgg_features = features.cpu().numpy()
    attributes = attributes.cpu().numpy()
    texts = texts.cpu().numpy()
    labels = labels.cpu().numpy()
    
    model0 = model0.cpu()
    torch.cuda.empty_cache()
    
    print('Generate similarity matrix...very slow...')
    S_numpy, I_numpy = distance_fusion(vgg_features, texts, alpha, beta, labels, near_neighbor,a1)
    
    S_numpy = S_numpy.astype(np.float)
    
    S_buffer = torch.FloatTensor(S_numpy).cpu()
    I_buffer = torch.FloatTensor(I_numpy).cpu()
    attributes_buffer = torch.FloatTensor(attributes).cpu()

    bestMAP = -1.0
    print('Start training...')
    model.train()
    img_encoder.train()
    txt_encoder.train()
    n_batch = len(train_dataloader)
    
    jilu_xinxi_img = []
    jilu_xinxi_txt = []
    jilu_xinxi_tea = []
    
    jilu_zengqiang = []
    
    jilu_loss = []
    
    for epoch in range(resume_it, max_iter):
        for i, (data, text, _, index, imgx) in enumerate(train_dataloader):

            data = data.to(device)
            text = text.to(device).float()
            imgx = imgx.to(device).float()
            index = index.to(device)
            
            attribute = attributes_buffer[index, :]
            targets = S_buffer[index, :][:, index]
            
            attribute = attribute.to(device).float()
            targets = targets.to(device).float()

            cur_tea, cur_att, cur_code = model(data)
            cur_f = img_encoder(imgx)
            cur_g = txt_encoder(text)

            batch_size = data.size(0)
            I = torch.eye(batch_size).cuda()
            
            sim_loss = tri_loss(cur_f, cur_g, cur_tea, targets, I, ceshi_tri,ceshi_tea)

            code_loss = torch.sum(torch.pow(cur_code - cur_tea, 2))
            att_loss = torch.sum(torch.pow(cur_att - attribute, 2))

            
            att_loss = (code_loss+att_loss)/(batch_size)

            sf_m = mutual_information(targets, cur_f, 1.0, 1.0, 2.0)
            sg_m = mutual_information(targets, cur_g, 1.0, 1.0, 2.0)
            st_m = mutual_information(targets, cur_tea, 1.0, 1.0, 2.0)
            
            if i % 50 == 0:
                jilu_xinxi_img.append(sf_m)
                jilu_xinxi_txt.append(sg_m)
                jilu_xinxi_tea.append(st_m)
                

            if sf_m > sg_m:
                simfen_loss = cross_loss(cur_tea, cur_g, targets, I)
                if i % 50 == 0:
                    jilu_zengqiang.append(0)
            else:
                simfen_loss = cross_loss(cur_f, cur_tea, targets, I)
                if i % 50 == 0:
                    jilu_zengqiang.append(1)
            
            loss = sim_loss + gamma*att_loss + ceshi_cross*simfen_loss

            optimizer_img.zero_grad()
            optimizer_txt.zero_grad()
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_img.step()
            optimizer_txt.step()
            
            if i % 50 == 0:
                jilu_loss.append(loss.item())

            # Print log
            if verbose:
                logger.debug('[epoch:{}][Batch:{}/{}][loss:{:.4f}]'.format(epoch+1, i+1, n_batch, loss.item()))
        if epoch % 2 == 0:
            
            XXmAP = evaluate_tea(model, query_dataloader, retrieval_dataloader, code_length, label_dim, device, topk, multi_labels)
            logger.info('[iteration:{}][tea XXmap:{:.4f}]'.format(epoch + 1, XXmAP))
            XXmAP, YYmAP, XYmAP, YXmAP, Qi, Qt, Di, Dt, query_L, retrieval_L = evaluate(img_encoder,
                                                  txt_encoder,
                                                  query_dataloader,
                                                  retrieval_dataloader,
                                                  code_length,
                                                  label_dim,
                                                  device,
                                                  topk,
                                                  multi_labels,
                                                  'stu'
                                                  )

            logger.info('[iteration:{}][stu XXmap:{:.4f}]'.format(epoch + 1, XXmAP))
            logger.info('[iteration:{}][stu YYmap:{:.4f}]'.format(epoch + 1, YYmAP))
            logger.info('[iteration:{}][stu XYmap:{:.4f}]'.format(epoch + 1, XYmAP))
            logger.info('[iteration:{}][stu YXmap:{:.4f}]'.format(epoch + 1, YXmAP))
            
            fus_mAP = XYmAP + YXmAP
            
            if fus_mAP > bestMAP:
                bestMAP = fus_mAP
                Qi = Qi.cpu().numpy().astype(np.int)
                Qt = Qt.cpu().numpy().astype(np.int)
                Di = Di.cpu().numpy().astype(np.int)
                Dt = Dt.cpu().numpy().astype(np.int)
                query_L = query_L.cpu().numpy().astype(np.int)
                retrieval_L = retrieval_L.cpu().numpy().astype(np.int)
                

    checkpoint_name = arch + '_' + dataset + '_' + str(code_length)
    with open(checkpoint_name+'.pkl', 'wb') as output:
        data1 = {'jilu_xinxi_img':jilu_xinxi_img,
        'jilu_xinxi_txt':jilu_xinxi_txt,
        'jilu_xinxi_tea':jilu_xinxi_tea,
        'jilu_zengqiang':jilu_zengqiang,
        'jilu_loss':jilu_loss}
        pkl.dump(data1, output)
        

def evaluate_tea(model, query_dataloader, retrieval_dataloader, code_length, label_dim, device, topk, multi_labels):
    
    model.eval()

    # Generate hash code
    print('Generate Query Set Code...')
    #query_code, onehot_query_targets = generate_code(model, query_dataloader, code_length, label_dim, device)
    Xquery_code = generate_code_stu(model, query_dataloader, code_length, label_dim, device)
    print('Generate Retrieval Set Code...')
    #retrieval_code, onehot_retrieval_targets = generate_code(model, retrieval_dataloader, code_length, label_dim, device)
    Xretrieval_code = generate_code_stu(model, retrieval_dataloader, code_length, label_dim, device)
    
    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)

    
    # Calculate mean average precision
    XXmAP = mean_average_precision(
        Xquery_code,
        Xretrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    
    model.train()

    return XXmAP

def generate_code_stu(model, dataloader, code_length, label_dim, device):
    
    with torch.no_grad():
        N = len(dataloader.dataset)
        Xcode = torch.zeros([N, code_length])
        for data, text, label, index, imgx in tqdm(dataloader):
            data = data.to(device)
            outputs, _, _ = model(data)
            Xcode[index, :] = outputs.sign().cpu()
            
    return Xcode
    
def evaluate(model, txt_encoder, query_dataloader, retrieval_dataloader, code_length, label_dim, device, topk, multi_labels, leixing):
    """
    Evaluate.

    Args
        model(torch.nn.Module): CNN model.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.
        multi_labels(bool): Multi labels.

    Returns
        mAP(float): Mean average precision.
    """
    model.eval()

    # Generate hash code
    print('Generate Query Set Code...')
    #query_code, onehot_query_targets = generate_code(model, query_dataloader, code_length, label_dim, device)
    Xquery_code, Yquery_code, _ = generate_code(model, txt_encoder, query_dataloader, code_length, label_dim, device, leixing)
    print('Generate Retrieval Set Code...')
    #retrieval_code, onehot_retrieval_targets = generate_code(model, retrieval_dataloader, code_length, label_dim, device)
    Xretrieval_code, Yretrieval_code, _ = generate_code(model, txt_encoder, retrieval_dataloader, code_length, label_dim, device, leixing)
    
    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)

    # Calculate mean average precision
    XXmAP = mean_average_precision(
        Xquery_code,
        Xretrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    YYmAP = mean_average_precision(
        Yquery_code,
        Yretrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    XYmAP = mean_average_precision(
        Xquery_code,
        Yretrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    YXmAP = mean_average_precision(
        Yquery_code,
        Xretrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )

    model.train()

    return XXmAP, YYmAP, XYmAP, YXmAP, Xquery_code, Yquery_code, Xretrieval_code, Yretrieval_code, onehot_query_targets, onehot_retrieval_targets


def generate_code(model, txt_encoder, dataloader, code_length, label_dim, device, leixing):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        Xcode = torch.zeros([N, code_length])
        Ycode = torch.zeros([N, code_length])
        labels = torch.zeros([N, label_dim])
        for data, text, label, index, imgx in tqdm(dataloader):
            data = data.to(device)
            text = text.to(device).float()
            imgx = imgx.to(device).float()
            if leixing == 'tea':
                outputs, _, _ = model(data)
            else:
                outputs = model(imgx)
            outputsY = txt_encoder(text)
            Xcode[index, :] = outputs.sign().cpu()
            Ycode[index, :] = outputsY.sign().cpu()
            labels[index, :] = label.float()
        labels = labels.to(device)

    return Xcode, Ycode, labels

def distance_fusion(vgg_features, texts, alpha, beta, train_labels, near_neighbor,a1):

    Sbiao = np.matmul(train_labels, np.transpose(train_labels))
    Sbiao_true = (Sbiao > 0) * 1.0
    Sbiao_false = (Sbiao == 0) * -1.0
    Sbiao = (Sbiao_true + Sbiao_false)
    
    
    F_I = torch.FloatTensor(vgg_features).cuda()
    F_T = torch.FloatTensor(texts).cuda()
    F_I = F.normalize(F_I)
    F_T = F.normalize(F_T)
    
    print('Seek for VGG embedding...')
    vgg_dist = pdist(vgg_features, metric='cosine')
    vgg_dist = squareform(vgg_dist)
    Svgg = generate_similarity_matrix(vgg_dist, alpha, beta)
    print('Seek for Text embedding...')
    ptexts = pdist(texts, metric='euclidean')
    ptexts = squareform(ptexts)
    Stxt = np.zeros_like(ptexts)
    Stxt[ptexts<=1]=1
    Stxt[ptexts>=5]=-1
    Snear = 1.0 * (Svgg + Stxt == 2)
    Sfar = -1.0 * (((Svgg == -1) + (Stxt == -1)) > 0)
    S = Snear + Sfar
    I = np.zeros(np.shape(S))
    I = (((S == 1.0) + (S == -1.0)) == 1)
    S_I = F_I.mm(F_I.t())
    S_T = F_T.mm(F_T.t())
    S_pair = a1 * S_T + (1 - a1) * S_I
    S_pair = S_pair * 2.0 - 1
    S_pair = S_pair.cpu().numpy()
    S_pair[I==1] = 0
    conS = S_pair + S
    
    return conS, I

def generate_similarity_matrix(dist_matrix, alpha, beta):
    """
    Generate similarity matrix.

    Args
        features(torch.Tensor): Features.
        alpha, beta(float): Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """
    # Cosine similarity
    cos_dist = dist_matrix
    # Find maximum count of cosine distance
    max_cnt, max_cos = 0, 0
    interval = 1. / 100
    cur = 0
    for i in range(100):
        cur_cnt = np.sum((cos_dist > cur) & (cos_dist < cur + interval))
        if max_cnt < cur_cnt:
            max_cnt = cur_cnt
            max_cos = cur
        cur += interval

    # Split features into two parts
    flat_cos_dist = cos_dist.reshape((-1, 1))
    left = flat_cos_dist[np.where(flat_cos_dist <= max_cos)[0]]
    right = flat_cos_dist[np.where(flat_cos_dist > max_cos)[0]]

    # Reconstruct gaussian distribution
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([2 * max_cos - right, right])

    # Model data using gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    # Construct similarity matrix
    S = (cos_dist < (left_mean - alpha * left_std)) * 1.0 + (cos_dist > (right_mean + beta * right_std)) * -1.0
    # return torch.FloatTensor(S), torch.FloatTensor(I), torch.FloatTensor(D)
    return S

def extract_features(model, model0, dataloader, feature_dim, attribute_dim, device, verbose, label_dim, txt_dim):
    """
    Extract features.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        feature_dim(int): Number of features.
        device(torch.device): Using GPU or CPU.
        verbose(bool): Print log.

    Returns
        features(torch.Tensor): Features.
    """
    model.eval()
    model0.eval()
    model.set_extract_features(True)
    features = torch.zeros(len(dataloader.dataset.data[0]), 4096)
    texts = torch.zeros(len(dataloader.dataset.data[0]), txt_dim)
    labels = torch.zeros(len(dataloader.dataset.data[0]), label_dim).double()
    attributes = torch.zeros(len(dataloader.dataset.data[0]), attribute_dim)
    with torch.no_grad():
        N = len(dataloader)
        for i, (data, text, label, index, imgx) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i + 1, N))
            data = data.to(device)

            features[index, :] = imgx.float().cpu()
            attributes[index, :] = model0(data).cpu()
            labels[index, :] = label.double().cpu()
            texts[index, :] = text.float().cpu()

    model.set_extract_features(False)
    model.train()

    return features, texts, attributes, labels
