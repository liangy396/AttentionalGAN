from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data
from dataset_fashiongen2 import TextDataset as TextFashionGenDataset

from model import RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/fashiongen2.yml', type=str)
                        #default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0

    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

        words_features, sent_code = cnn_model(real_imgs[-1])
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels, cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step

    return s_cur_loss, w_cur_loss


def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    print('initialization finished')
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    print(cfg.CUDA)
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    # seeds
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    # parameters
    cfg.DATASET_NAME = 'fashiongen2'
    cfg.DATA_DIR = '../data/fashiongen'
    cfg.CONFG_NAME = 'DAMSM'
    cfg.TRAIN.BATCH_SIZE = 32
    #cfg.TRAIN.NET_E = '../DAMSMencoders/fashiongen2/text_encoder80.pth'
    #cfg.TRAIN.NET_E = '../output/fashiongen2__2019_11_23_19_50_01/Model/text_encoder50.pth'
    #cfg.TRAIN.NET_E = '../output/fashiongen2__2019_12_06_22_09_21/Model/text_encoder9.pth'
    cfg.TRAIN.NET_E = '../output/fashiongen2__2019_12_08_00_18_18/Model/text_encoder3.pth'
    cfg.TRAIN.MAX_EPOCH = 100
    cfg.TRAIN.SNAPSHOT_INTERVAL = 5
    cfg.TRAIN.ENCODER_LR = 0.0002
    cfg.TRAIN.RNN_GRAD_CLIP = 0.25
    cfg.TEXT.CAPTIONS_PER_IMAGE = 1
    cfg.TEXT.WORDS_NUM = 10
    cfg.TEXT.EMBEDDING_DIM = 256
    cfg.WORKERS = 1
    cfg.TRAIN.SMOOTH.GAMMA1 = 1.0 #4
    cfg.TRAIN.SMOOTH.GAMMA2 = 1.0  #5
    cfg.TRAIN.SMOOTH.GAMMA3 = 10.0
    

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    #model_dir = os.path.join(output_dir, 'Model')
    #image_dir = os.path.join(output_dir, 'Image')
    #mkdir_p(model_dir)
    #mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE

    print(imsize)
    print(batch_size)
    
    print(cfg.DATA_DIR)
    
    print(cfg.DATASET_NAME)

    # dataset images transforms
    if cfg.DATASET_NAME == 'fashiongen2':
        image_transform = transforms.Compose([
            transforms.Resize(imsize),
            transforms.RandomHorizontalFlip()])
    else:
        image_transform = transforms.Compose([
            transforms.Scale(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

    if cfg.DATASET_NAME == 'fashiongen2':
        dataset = TextFashionGenDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    #print(dataset.captions)
    print(cfg.TRAIN.MAX_EPOCH)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    if cfg.DATASET_NAME == 'fashiongen2':
        dataset_val = TextFashionGenDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
        print('correct dataset')
    else:
        dataset_val = TextDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    print('start training')
    #cfg.TRAIN

    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR*pow(0.98,3)
        if len(dataloader_val) > 0:
        #if len(dataloader) > 0:
            #s_loss, w_loss = evaluate(dataloader, image_encoder,
            #                          text_encoder, batch_size)
            s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                      text_encoder, batch_size)
            print('| valid loss '
                  '{:5.2f} {:5.2f} | lr {:.5f}|'
                  .format(s_loss, w_loss, lr))
        print('-' * 89)
        if lr > cfg.TRAIN.ENCODER_LR/10.:
            lr *= 0.98

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
