import torch
import time
from torch.autograd import Variable
import pandas as pd
import os
import numpy as np
# Make a dictionary
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    def add_word(self,word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
    def load_txt(self,path):
        train_data = pd.read_csv(path,sep="\t")
        Length = len(train_data)
        for index in range(Length):
            sentA = train_data.iloc[index]['sentence_A'].split()
            for word in sentA:
                self.add_word(word)
            sentB = train_data.iloc[index]['sentence_B'].split()
            for word in sentB:
                self.add_word(word)
    def __len__(self):
        return len(self.idx2word)
# sentence to index sequence
def seq_to_index(sentence,word_to_index):
    sentence = sentence.split()
    ids = [word_to_index[word] for word in sentence]
    tensor = torch.LongTensor(ids)
    return Variable(tensor)
# training process
def train_model(model,train_data,test_data,dictionary_data,optimizer,loss_function,params):
    epoches = params['epoches']
    Length = len(train_data)
    all_losses = []
    peason_list = []
    mse_list = []
    for epoch in range(epoches):
        for index in range(Length):
            sentA = seq_to_index(train_data.iloc[index]['sentence_A'],dictionary_data.word2idx)
            sentB = seq_to_index(train_data.iloc[index]['sentence_B'],dictionary_data.word2idx)
            score = (float(train_data.iloc[index]['relatedness_score']) - 1)/4
            target_score = Variable(torch.FloatTensor([score]).view(1, 1))
            sentA = sentA.cuda()
            sentB = sentB.cuda()
            target_score = target_score.cuda()
            optimizer.zero_grad()
            hidden = model.initial_hidden()
            predict_score = model(sentA,sentB,hidden)
            loss = loss_function(predict_score,target_score)
            loss.backward()
            optimizer.step()
            all_losses.append(loss.data[0])
            if index %100 == 0:
                print("index == %d"%index)
        # save model
        model_file = params['model_name']+ ".model"
        model_name = params['model_name']
        torch.save(model,model_file)
        # test Data
        model = torch.load(model_file)
        test_data = pd.read_csv(os.path.join('Data','test.txt'),sep ="\t")
        x_data ,y_data = Test_data(model,test_data,dictionary_data)
        PeaR = Peason(x_data,y_data)
        MseR = MSE_Loss(x_data,y_data)
        peason_list.append(PeaR)
        mse_list.append(MseR)
        if not os.path.exists(model_file):
            os.mkdir(model_name)
        with open(model_name+"%d_Y.txt"%epoch,"w") as file:
            file.write(str(y_data))
        print("epoches:%d\t MES:%f\t Peason:%f \t"%(epoch,MseR,PeaR))
    return all_losses,peason_list,mse_list
def Test_data(model,test_data,dictionary_data):
    Length = len(test_data)
    x_data = []
    y_data = []
    model.cuda()
    for index in range(Length):
        sentA = seq_to_index(test_data.iloc[index]['sentence_A'],dictionary_data.word2idx)
        sentB = seq_to_index(test_data.iloc[index]['sentence_B'],dictionary_data.word2idx)
        score = (float(test_data.iloc[index]['relatedness_score']) - 1)/4
        target_score = Variable(torch.FloatTensor([score]).view(1, 1))
        sentA = sentA.cuda()
        sentB = sentB.cuda()
        target_score = target_score.cpu()
        hidden = model.initial_hidden()
        predict_score = model(sentA,sentB,hidden)
        predict_score = predict_score.cpu()
        x_data.append(target_score.data[0])
        y_data.append(predict_score.data[0])
        if (index%100==0):
            print("processed data:%d"%index)
    return np.array(x_data),np.array(y_data)
def Peason(x,y):
    x = x - x.mean()
    y = y - y.mean()
    return x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
def MSE_Loss(x,y):
    Tmp = x - y
    Tmp = np.power(Tmp,2)
    return Tmp.mean()