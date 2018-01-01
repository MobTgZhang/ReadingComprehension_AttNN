import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import AttSimpleRNN
from manager_torch import GPUManager
from utils import Dictionary
from utils import train_model,Test_data,Peason,MSE_Loss
if __name__ == '__main__':
    dictionary_data = Dictionary()
    dictionary_data.load_txt(os.path.join('Data','train.txt'))
    dictionary_data.load_txt(os.path.join('Data','test.txt'))
    params = {
        'vocab_size': len(dictionary_data),
        'embedding_dim': 300,
        'hidden_dim': 200,
        'temp_dim':150,
        'out_dim': 1,
        'algorithm': 'gru',
        'learning_rate':1e-3,
        'epoches':50,
        'model_name':'AttSimpleRNN'
    }
    algorithm = ['gru','lstm','rnn']
    test_data = pd.read_csv(os.path.join('Data','test.txt'),sep = "\t")
    for k in range(len(algorithm)):
        params['algorithm'] = algorithm[k]
        params['model_name'] = params['model_name']+params['algorithm']
        model_file = params['model_name'] + ".model"
        model_name = params['model_name']
        train_data = pd.read_csv(os.path.join('Data','train.txt'),sep = "\t")
        if not os.path.exists(model_file):
            gm = GPUManager()
            torch.cuda.set_device(gm.auto_choice())
            Model = AttSimpleRNN(params)
            optimizer = optim.Adam(Model.parameters(),params['learning_rate'])
            loss_function= nn.BCELoss()
            all_losses,peason_list,mse_list = train_model(Model,train_data,test_data,dictionary_data,optimizer,loss_function,params)
            with open(model_name+"loss.txt","w") as file:
                for k in range(len(all_losses)):
                    file.write(str(all_losses[k]) + "\n")
            with open(model_name+"Peason.txt","w") as file:
                for k in range(len(peason_list)):
                    file.write(str(peason_list[k]) + "\n")
            with open(model_name+"MSE_Loss.txt","w") as file:
                for k in range(len(mse_list)):
                    file.write(str(mse_list[k]) + "\n")
            print("Model :%s Completed!"%model_name)
        # test Data
        model = torch.load(model_file)
        x_data ,y_data = Test_data(model,test_data,dictionary_data)
        resultA = Peason(x_data,y_data)
        resultB = MSE_Loss(x_data,y_data)
        print("Test Model:%s Completed! Peason:%f MSE:%f"%(model_name,resultA,resultB))
        with open(model_name+"_peason.txt","w") as file:
            file.write(str(resultA) + "\t" + str(resultB))
        with open(model_name+"_Y.txt","w") as file:
            file.write(str(y_data))