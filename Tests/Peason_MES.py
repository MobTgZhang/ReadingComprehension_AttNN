import numpy as np
import os
import pandas as pd
def Peason(x,y):
    x = x - x.mean()
    y = y - y.mean()
    return x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
def MSE_Loss(x,y):
    Tmp = x - y
    Tmp = np.power(Tmp,2)
    return Tmp.mean()
def get_data(file_path):
	result = []
	with open(file_path,"r") as file:
		line = file.read()
		line = line[1:len(line)-1]
		result_list = line.split()
		for item in result_list:
			result.append(float(item))
	return np.array(result)
def main():
	AttSimpleRNN_RNN = "AttSimpleRNNgru_Y.txt"
	AttSimpleRNN_GRU = "AttSimpleRNNlstm_Y.txt"
	AttSimpleRNN_LSTM = "AttSimpleRNNrnn_Y.txt"
	SimpleRNN_GRU = "SimpleRNNlstm_Y.txt"
	SimpleRNN_LSTM = "SimpleRNNgru_Y.txt"
	path = "LDA_LSI_TF-IDF"
	LDA = "predict_lda.txt"
	LSI = "predict_lsi.txt"
	TD_IDF = "predict_lsi.txt"
	labeled = "labeled.txt"

	resultA = get_data(AttSimpleRNN_RNN)
	resultB = get_data(AttSimpleRNN_GRU)
	resultC = get_data(AttSimpleRNN_LSTM)
	resultD = get_data(SimpleRNN_GRU)
	resultE = get_data(SimpleRNN_LSTM)
	resultF = get_data(os.path.join(path,LDA))
	resultG = get_data(os.path.join(path,LSI))
	resultH = get_data(os.path.join(path,TD_IDF))
	data = pd.read_csv(os.path.join(path,labeled),sep = "\t")
	labeled_e = []
	for index in range(len(data)):
		labeled_e.append(float(data.iloc[index]['relatedness_score']))
	labeled_e = np.array(labeled_e)
	result = [resultA,resultB,resultC,resultD,resultE,resultF,resultG,resultH]
	peason = []
	mseloss = []
	name = ["Att-RNN","Att-GRU","Att-LSTM","SimpleRNN+GRU","SimpleRNN+LSTM","LDA","LSI","TF-IDF"]
	print("Peason\tMSE_Loss")
	for item in result:
		print(Peason(item,labeled_e),MSE_Loss(item,labeled_e))
		peason.append(Peason(item,labeled_e))
		mseloss.append(MSE_Loss(item,labeled_e))
if __name__ == '__main__':
	main()