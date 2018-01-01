import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
def get_data(file_path):
	result = []
	with open(file_path,"r") as file:
		while True:
			line = file.readline()
			if not line:
				break
			line = float(line)
			result.append(line)
	return result
def get_packed_data(result,epoches):
	data = np.array(result)
	batch_size = data.shape[0]//epoches
	y = data.reshape(50,batch_size)
	y = np.mean(y,axis = 1)
	return y
# draw the picture of the all model
def drawPic():
	AttSimpleRNN_RNN = "AttSimpleRNNrnnloss.txt"
	AttSimpleRNN_GRU = "AttSimpleRNNlstmloss.txt"
	AttSimpleRNN_LSTM = "AttSimpleRNNgruloss.txt"
	SimpleRNN_GRU = "SimpleRNNgruloss.txt"
	SimpleRNN_LSTM = "SimpleRNNlstmloss.txt"
	resultA = get_data(AttSimpleRNN_RNN)
	resultB = get_data(AttSimpleRNN_GRU)
	resultC = get_data(AttSimpleRNN_LSTM)
	resultD = get_data(SimpleRNN_GRU)
	resultE = get_data(SimpleRNN_LSTM)
	y1 = get_packed_data(resultA,50)
	y2 = get_packed_data(resultB,50)
	y3 = get_packed_data(resultC,50)
	y4 = get_packed_data(resultD,50)
	y5 = get_packed_data(resultE,50)
	x = np.linspace(0,50,50)
	plt.plot(x,y1,label = "Att-RNN")
	plt.plot(x,y2,label = "Att-GRU")
	plt.plot(x,y3,label = "Att-LSTM")
	plt.plot(x,y4,label = "SimpleGRU")
	plt.plot(x,y5,label = "SimpleLSTM")
	plt.title("The Loss of model")
	plt.ylabel("Loss")
	plt.xlabel("Trainig batches")
	plt.legend()
	plt.show()
# get the score of the length to the sentence
# relatedness , predict , sentenceLength 
def getLength(labeled_file):
	data_all = pd.read_csv(labeled_file,sep = "\t")
	sentence_all = []
	labeled = []
	Length = len(data_all)
	MaxLength = 0
	for index in range(Length):
		sentA = data_all.iloc[index]['sentence_A'].split()
		sentB = data_all.iloc[index]['sentence_B'].split()
		score = float(data_all.iloc[index]['relatedness_score'])
		labeled.append(score)
		temp = int((len(sentA)+len(sentB))/2)
		MaxLength = max(len(sentA),len(sentB),MaxLength,temp)
		sentence_all.append(temp)
	return np.array(labeled,dtype= np.float32),sentence_all,MaxLength
def get_model(model_name):
	if model_name == "SimpleGRU":
		predict_file = "SimpleRNNgru_Y.txt"
	elif model_name == "SimpleLSTM":
		predict_file = "SimpleRNNlstm_Y.txt"
	elif model_name == "Att-RNN":
		predict_file = "AttSimpleRNNrnn_Y.txt"
	elif model_name == "Att-GRU":
		predict_file = "AttSimpleRNNgru_Y.txt"
	elif model_name == "Att-LSTM":
		predict_file = "AttSimpleRNNlstm_Y.txt"
	else:
		raise Exception("Unknown model: " + model_name)
	with open(predict_file,"r") as file:
		line = file.read()
		line = line[1:len(line)-1]
		result = line.split()
		all_data = []
		for item in result:
			all_data.append(float(item))
	return np.array(all_data)
def draw_My_Pic():
	# To test the length of the sentence to set the sentences
	path = "LDA_LSI_TF-IDF"
	labeled = "labeled.txt"
	labeled_file = os.path.join(path,labeled)
	model_name = ['Att-GRU','SimpleGRU','Att-RNN','Att-LSTM','SimpleLSTM']
	colors = ['red','green','blue','yellow','pink']
	labeled,sentenceLength,MaxLength = getLength(labeled_file)
	length = np.zeros((MaxLength,),dtype = np.float32)
	num = np.zeros((MaxLength,),dtype = np.int)
	plt.figure()
	# fig.suptitle(dname,fontsize=16,x=0.53,y=1.05,)
	plt.title("The Influence of Sentence Length on Sentence Similarity")
	for index in range(len(model_name)):
		predict = get_model(model_name[index])
		#print(len(predict),len(sentenceLength))
		for k in range(len(predict)):
			length[sentenceLength[k]-1] += predict[k]
			num[sentenceLength[k]-1] += 1
		for k in range(len(length)):
			if num[k]!=0:
				length[k] = length[k]/num[k]
		y = length
		x = np.linspace(0,len(y),len(y))
		plt.subplot(2,3,index+1)
		plt.bar(x,y,label = model_name[index],color = colors[index])
		plt.legend()
	plt.show()
def main():
	drawPic()
	draw_My_Pic()
if __name__ == '__main__':
	main()