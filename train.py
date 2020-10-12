import os
import sys
import glob
import numpy as np
import argparse
import asyncio
import kaldi
import time
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict, Counter

from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

from joblib import Parallel, delayed


def schedule_lr(optimizer, factor=0.1):
    for params in optimizer.param_groups:
        params['lr'] *= factor

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def read_mat(name):
    feat = kaldi.read_mat(name).numpy()
    return feat 


def load_n_col(file, numpy=False):
    data = []
    with open(file) as fp:
        for line in fp:
            data.append(line.strip().split(' '))
    columns = list(zip(*data))
    if numpy:
        columns = [np.array(list(i)) for i in columns]
    else:
        columns = [list(i) for i in columns]
    return columns


def odict_from_2_col(file, numpy=False):
    col0, col1 = load_n_col(file, numpy=numpy)
    return OrderedDict({c0: c1 for c0, c1 in zip(col0, col1)})

def load_one_tomany(file, numpy=False):
    one = []
    many = []
    with open(file) as fp:
        for line in fp:
            line = line.strip().split(' ', 1)
            one.append(line[0])
            m = line[1].split(' ')
            many.append(np.array(m) if numpy else m)
    if numpy:
        one = np.array(one)
    return one, many


def train_transform(feats, seqlen):
    leeway = feats.shape[0] - seqlen
    startslice = np.random.randint(0, int(leeway)) if leeway > 0  else 0
    feats = feats[startslice:startslice+seqlen] if leeway > 0 else np.pad(feats, [(0,-leeway), (0,0)], 'constant')
    return torch.FloatTensor(feats)


async def get_item_train(instructions):
    fpath = instructions[0]
    seqlen = instructions[1]
    raw_feats = read_mat(fpath)
    feats = train_transform(raw_feats, seqlen)
    return feats


async def get_item_test(filepath):
    raw_feats = read_mat(filepath)
    return torch.FloatTensor(raw_feats)


def async_map(coroutine_func, iterable):
    loop = asyncio.get_event_loop()
    future = asyncio.gather(*(coroutine_func(param) for param in iterable))
    return loop.run_until_complete(future)


class SpeakerDataset(Dataset):
    def __init__(self, data_base_path, asynchr=False, num_workers=5):
        self.data_base_path = data_base_path
        self.num_workers = num_workers
        utt2spk_path = os.path.join(data_base_path, 'utt2spk')
        spk2utt_path = os.path.join(data_base_path, 'spk2utt')
        feats_scp_path = os.path.join(data_base_path, 'feats.scp')

        assert os.path.isfile(utt2spk_path)
        assert os.path.isfile(feats_scp_path)
        assert os.path.isfile(spk2utt_path)

        self.utts, self.uspkrs = load_n_col(utt2spk_path)
        self.utt_fpath_dict = odict_from_2_col(feats_scp_path)

        self.label_enc = LabelEncoder()

        self.spkrs, self.spkutts = load_one_tomany(spk2utt_path)
        self.spkrs = self.label_enc.fit_transform(self.spkrs)
        self.spk_utt_dict = OrderedDict({k:v for k,v in zip(self.spkrs, self.spkutts)})

        self.uspkrs = self.label_enc.transform(self.uspkrs)
        self.utt_spkr_dict = OrderedDict({k:v for k,v in zip(self.utts, self.uspkrs)})

        self.utt_list = list(self.utt_fpath_dict.keys())
        self.first_batch = True

        self.num_classes = len(self.label_enc.classes_)
        self.asynchr = asynchr

        self.allowed_classes = self.spkrs.copy() # classes the data can be drawn from
        self.idpool = self.allowed_classes.copy()
        self.ignored = []

    def __len__(self):
        return len(self.utt_list)


    @staticmethod
    def get_item(instructions):
        fpath = instructions[0]
        seqlen = instructions[1]
        feats = read_mat(fpath)
        feats = train_transform(feats, seqlen)
        return feats


    def set_remaining_classes(self, remaining:list):
        self.allowed_classes = sorted(list(set(remaining)))
        self.ignored = sorted(set(np.arange(self.num_classes)) - set(remaining))
        self.idpool = self.allowed_classes.copy()

    def set_ignored_classes(self, ignored:list):
        self.ignored = sorted(list(set(ignored)))
        self.allowed_classes = sorted(set(np.arange(self.num_classes)) - set(ignored))
        self.idpool = self.allowed_classes.copy()

    def set_remaining_classes_comb(self, remaining:list, combined_class_label):
        remaining.append(combined_class_label)
        self.allowed_classes = sorted(list(set(remaining)))
        self.ignored = sorted(set(np.arange(self.num_classes)) - set(remaining))
        self.idpool = self.allowed_classes.copy()        
        for ig in self.ignored:
            # modify self.spk_utt_dict[combined_class_label] to contain all the ignored ids utterances
            self.spk_utt_dict[combined_class_label] += self.spk_utt_dict[ig]
        self.spk_utt_dict[combined_class_label] = list(set(self.spk_utt_dict[combined_class_label]))


    def get_batches(self, batch_size=256, max_seq_len=400):
        # with Parallel(n_jobs=self.num_workers) as parallel:
        assert batch_size < len(self.allowed_classes) #Metric learning assumption large num classes
        lens = [max_seq_len for _ in range(batch_size)]
        while True:

            if len(self.idpool) <= batch_size:
                batch_ids = np.array(self.idpool)
                self.idpool = self.allowed_classes.copy()
                rem_ids = np.random.choice(self.idpool, size=batch_size-len(batch_ids), replace=False)
                batch_ids = np.concatenate([batch_ids, rem_ids])
                self.idpool = list(set(self.idpool) - set(rem_ids))
            else:
                batch_ids = np.random.choice(self.idpool, size=batch_size, replace=False)
                self.idpool = list(set(self.idpool) - set(batch_ids))

            batch_fpaths = []
            for i in batch_ids:
                utt = np.random.choice(self.spk_utt_dict[i])
                batch_fpaths.append(self.utt_fpath_dict[utt])


            if self.asynchr:
                batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            else:
                #batch_feats = [self.get_item(a) for a in zip(batch_fpaths, lens)]
                batch_feats = Parallel(n_jobs=self.num_workers)(delayed(self.get_item)(a) for a in zip(batch_fpaths, lens))



            yield torch.stack(batch_feats), list(batch_ids)


class XVecHead(nn.Module):
    def __init__(self, num_features, num_classes, hidden_features=None):
        super(XVecHead, self).__init__()
        hidden_features = num_features if not hidden_features else hidden_features
        self.fc_hidden = nn.Linear(num_features, hidden_features)
        self.nl = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(hidden_features)
        self.fc = nn.Linear(hidden_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, input):
        input = self.fc_hidden(input)
        input = self.nl(input)
        input = self.bn(input)
        W = self.fc.weight
        b = self.fc.bias
        logits = F.linear(input, W, b)
        return logits


class TDNN(nn.Module):
    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1, padding=0, dropout_p=0.1):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding

        self.kernel = nn.Conv1d(self.input_dim, self.output_dim, self.context_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
        self.nonlinearity = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = self.kernel(x.transpose(1,2))
        x = self.kernel(x)
        x = self.nonlinearity(x)
        x = self.drop(x)
        x = self.bn(x)
        return x.transpose(1,2)


class StatsPool(nn.Module):
    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel
    
    def forward(self, x):
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x


class ETDNN(nn.Module):
    def __init__(self, features_per_frames=60, hidden_features=1024, embed_features=512, num_classes=7000):
        super(ETDNN, self).__init__()

        self.features_per_frames = features_per_frames
        self.hidden_features = hidden_features
        self.embed_features = embed_features
        self.num_classes = num_classes

        self.frame1 = TDNN(input_dim=self.features_per_frames, output_dim=self.hidden_features, context_size=5, dilation=1)
        self.frame2 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=1, dilation=1)
        self.frame3 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=3, dilation=2)
        self.frame4 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=1, dilation=1)
        self.frame5 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=3, dilation=3)
        self.frame6 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=1, dilation=1)
        self.frame7 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=3, dilation=4)
        self.frame8 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=1, dilation=1)
        self.frame9 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features*3, context_size=1, dilation=1)

        self.tdnn_list = nn.Sequential(self.frame1, self.frame2, self.frame3, self.frame4, self.frame5, self.frame6, self.frame7, self.frame8, self.frame9)

        self.statspool = StatsPool()

        self.fc_embed = nn.Linear(self.hidden_features*6, self.embed_features)
        self.norm_embed = torch.nn.BatchNorm1d(self.embed_features)

        self.xvechead = XVecHead(self.embed_features, self.num_classes, 512)

    def forward(self, x):
        x = self.tdnn_list(x)

        x = self.statspool(x)

        x = self.fc_embed(x)
        x = self.activation_embed(x)
        x = self.norm_embed(x)

        x = self.xvechead(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data", type=str, default="data/train/")
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--multi_gpu", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=400)
    parser.add_argument("--num_iterations", type=int, default=200000)
    parser.add_argument("--model_dir", type=str, default="output/")

    args = parser.parse_args()
    return args


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = not args.use_cuda and torch.cuda.is_available()
    print('='*30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('='*30)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = ETDNN()
    model = nn.DataParallel(model, device_ids=[0])
    model.to(f'cuda:{model.device_ids[0]}')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    ds_train = SpeakerDataset(args.data)
    data_generator = ds_train.get_batches(batch_size=args.batch_size, max_seq_len=args.max_seq_len)

    running_loss = [np.nan for _ in range(500)]

    model.train()

    scheduler_steps = [50000, 80000, 110000, 140000, 170000, 190000]
    scheduler_lambda = 0.5

    for iterations in range(1, args.num_iterations + 1):
        if iterations in scheduler_steps:
            schedule_lr(optimizer, factor=scheduler_lambda)

        feats, iden = next(data_generator)

        feats = feats.to(f'cuda:{model.device_ids[0]}')
        iden = torch.LongTensor(iden).to(f'cuda:{model.device_ids[0]}')

        preds = model(feats)
        loss = criterion(preds, iden)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.pop(0)
        running_loss.append(loss.item())
        rmean_loss = np.nanmean(np.array(running_loss))

        if iterations % 100 == 0:
            torch.save(model, args.model_dir+"/optimizer"+str(iterations)+".mat")
            msg = "{}: {}: [{}/{}] \t C-Loss:{:.4f}, AvgLoss:{:.4f}, lr: {}, bs: {}".format(args.model_dir, time.ctime(), iterations, args.num_iterations, loss.item(), rmean_loss, get_lr(optimizer), len(feats))
            print(msg)


if __name__ == '__main__':
    args = parse_args()
    train(args)

