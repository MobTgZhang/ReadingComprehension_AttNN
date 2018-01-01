import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import copy
import math

class Double_Linear(nn.Module):
    def __init__(self,in_dimA,in_dimB,out_feature,bias = True):
        super(Double_Linear,self).__init__()
        self.in_dimA = in_dimA
        self.in_dimB = in_dimB
        self.out_feature = out_feature
        self.weightA = Parameter(torch.randn(in_dimA,out_feature))
        self.weightB = Parameter(torch.randn(in_dimB,out_feature))
        if bias:
            self.bias = Parameter(torch.randn(out_feature))
        else:
            self.reset_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        stdvA = 1./math.sqrt(self.weightA.size(1))
        stdvB = 1./math.sqrt(self.weightB.size(1))
        self.weightA.data.uniform_(-stdvA,stdvA)
        self.weightB.data.uniform_(-stdvB,stdvB)
        if self.bias is not None:
            self.bias.data.uniform_(-stdvB,stdvB)
    def forward(self,inputA,inputB):
        return F.linear(inputA,self.weightA.t(),None)+F.linear(inputB,self.weightB.t(),self.bias)
    def __repr__(self):
        return self.__class__.__name__ + ' ( ('\
            + str(self.in_dimA) + ' , '\
            + str(self.in_dimB) + ' ) ->'\
            + str(self.out_feature) + ' ) '
# Simple RNN model with no attention
class SimpleRNN(nn.Module):
    def __init__(self,params,emb_matrix = None):
        super(SimpleRNN,self).__init__()
        # parameters for model
        self.vocab_size = params['vocab_size']
        self.embedding_dim =  params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.out_dim = params['out_dim']
        self.algorithm = params['algorithm'].lower()
        # embedding layers
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if emb_matrix is not None:
            self.embedding.weight.data = emb_matrix
        # RNN layer
        if self.algorithm == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers=1,
                               dropout=0.5,bidirectional=False,batch_first=True)
        elif self.algorithm == "rnn":
            self.rnn = nn.RNN(self.embedding_dim,self.hidden_dim,num_layers=1,
                              dropout=0.5,bidirectional=False,batch_first=True)
        elif self.algorithm == "gru":
            self.rnn = nn.GRU(self.embedding_dim,self.hidden_dim,num_layers=1,
                              dropout=0.5,bidirectional=False,batch_first=True)
        else:
            raise (NameError,"Unknown Type:" + self.algorithm)
        # Bi-Linear
        self.bilinear = Double_Linear(self.hidden_dim,self.hidden_dim,self.hidden_dim)
        # out Layer
        self.out = nn.Linear(self.hidden_dim,self.out_dim)
        # sigmoid layer
        self.sigmoid = nn.Sigmoid()
        # initialize weights
        self.initial_weights()
        self.cuda()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)

        _,hidA = self.rnn(embA)
        _,hidB = self.rnn(embB)
        if self.algorithm == 'lstm':
            hidA = hidA[0]
            hidB = hidB[0]
        hidA = hidA.squeeze().view(1,hidA.size(2))
        hidB = hidB.squeeze().view(1,hidB.size(2))
        biout = self.bilinear(hidA,hidB)
        out_l = self.out(F.tanh(biout))
        return self.sigmoid(out_l)
    def initial_hidden(self):
        if self.algorithm == 'lstm':
            return (Variable(torch.zeros(1,1,self.hidden_dim)),
                    Variable(torch.zeros(1,1,self.hidden_dim)))
        elif self.algorithm == 'gru' or self.algorithm == 'rnn':
            return Variable(torch.zeros(1,1,self.hidden_dim))
        else:
            raise (NameError, "Unknown Type:" + self.algorithm)
    def initial_weights(self):
        initrange = 0.1
        initrange_w = 0.5
        self.bilinear.weightA.data.uniform_(-initrange,initrange)
        self.bilinear.weightB.data.uniform_(-initrange,initrange)
        self.out.weight.data.uniform_(-initrange,initrange)
# Simple RNN with attention
class AttSimpleRNN(nn.Module):
    def __init__(self,params,emb_matrix = None):
        super(AttSimpleRNN,self).__init__()
        # parameters for model
        self.vocab_size = params['vocab_size']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.temp_dim = params['temp_dim']
        self.out_dim = params['out_dim']
        self.algorithm = params['algorithm'].lower()

        # layer for the RNN
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if emb_matrix is not None:
            self.embedding.weight.data = emb_matrix
        # RNN layer
        if self.algorithm == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers=1,
                               dropout=0.5,bidirectional=False,batch_first=True)
        elif self.algorithm == "rnn":
            self.rnn = nn.RNN(self.embedding_dim,self.hidden_dim,num_layers=1,
                              dropout=0.5,bidirectional=False,batch_first=True)
        elif self.algorithm == "gru":
            self.rnn = nn.GRU(self.embedding_dim,self.hidden_dim,num_layers=1,
                              dropout=0.5,bidirectional=False,batch_first=True)
        else:
            raise (NameError,"Unknown Type:" + self.algorithm)
        # bi-linear
        self.bi_linear = Double_Linear(self.hidden_dim,self.hidden_dim,self.temp_dim)
        # tanh layer
        self.tanh_layer = nn.Tanh()
        # temp hidden layer
        self.linear = nn.Linear(self.temp_dim,self.out_dim)
        # sigmoid layer
        self.sigmoid = nn.Sigmoid()
        # initialize the hidden layer weights
        self.initial_weights()
        self.cuda()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)

        # hidden layer for rnn
        _,hidA = self.rnn(embA,hidden)
        _,hidB = self.rnn(embB,hidden)
        if self.algorithm == 'lstm':
            hidA = hidA[0]
            hidB = hidB[0]
        # bi-Linear
        hidA = hidA.squeeze().view(1, hidA.size(2))
        hidB = hidB.squeeze().view(1, hidB.size(2))

        ht = hidA * hidB
        hv = torch.abs(hidA - hidB)
        ht = self.bi_linear(ht,hv)

        hv = self.tanh_layer(ht)
        ht = self.linear(hv)
        return self.sigmoid(ht)

    def initial_hidden(self):
        if self.algorithm == 'lstm':
            return (Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
        elif self.algorithm == 'gru' or self.algorithm == 'rnn':
            return Variable(torch.zeros(1, 1, self.hidden_dim)).cuda()
        else:
            raise (NameError, "Unknown Type:" + self.algorithm)
    def initial_weights(self):
        initrange = 0.1
        # bi-linear
        self.bi_linear.weightA.data.uniform_(-initrange,initrange)
        self.bi_linear.weightB.data.uniform_(-initrange,initrange)
        # temp hidden layer
        self.linear.weight.data.uniform_(-initrange,initrange)
