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

        return x




def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data", type=str, default="data/train/")
    parser.add_argument("--model", type=str, default="output.raw")
    parser.add_argument("--xvector", type=str, default="directory")

    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda")

    model = ETDNN()
    resume = torch.load(args.model)
    model.load_state_dict(torch.load(args.model, map_location={"cuda" : "cpu"}).module.state_dict())
    model.to(device)
    model.train(False)
    model.eval()

    rspecifier="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:{}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:{}/vad.scp ark:- |".format(args.data, args.data)

    wspecifier = "ark,scp:{}/xvector.ark,{}/xvector.scp".format(args.xvector, args.xvector)
    writer = kaldi.VectorWriter(wspecifier)

    reader = kaldi.SequentialMatrixReader(rspecifier)
    with torch.no_grad():
        for key, value in reader:
            torch.no_grad()
            value = torch.Tensor([value.numpy()]).to(device)
            emb = model(value)
            kemb = emb.cpu().detach().numpy() 
            writer.Write(key=key, value=kemb[0])
    reader.Close()
    writer.Close()
   


if __name__ == '__main__':
    args = parse_args()
    train(args)

