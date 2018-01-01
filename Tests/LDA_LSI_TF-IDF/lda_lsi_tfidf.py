import pandas as pd
import numpy as np
from gensim import models,corpora,similarities
import copy
import os
import pprint
# 模型的基本架构
class traditional_Model:
    def __init__(self):
        self.documents = None
        self.dictionary = None
        self.corpus = None
    def load_documents(self,test_data):
        self.documents = pd.read_csv(test_data,sep="\t")
        corpora_documents = []
        Length = len(self.documents)
        for index in range(Length):
            # 分词处理
            sentA = self.documents.iloc[index]['sentence_A'].lower().split()
            sentB = self.documents.iloc[index]['sentence_B'].lower().split()
            corpora_documents.append(sentA)
            corpora_documents.append(sentB)
        self.dictionary = corpora.Dictionary(corpora_documents)
        self.corpus = [self.dictionary.doc2bow(text) for text in corpora_documents]
    def cal_related_ness(self,method):
        if method == "lda":
            return self._lda("predict_lda.txt")
        elif method == "lsi":
            return self._lsi("predict_lsi.txt")
        elif method == "tf-idf":
            self._tfidf("predict_tf_idf.txt")
        else:
            raise Exception("Unknow method: " + method)
    def _lda(self,save_file):
        with open(save_file,"w") as file:
            Length = len(self.documents)
            for index in range(Length):
                sentA = self.documents.iloc[index]['sentence_A'].lower().split()
                sentB = self.documents.iloc[index]['sentence_B'].lower().split()
                score = self.cos_sim_lda(sentA,index)
                file.write(str(score) + "\n")
                print(score)
    def _lsi(self,save_file):
        with open(save_file,"w") as file:
            Length = len(self.documents)
            for index in range(Length):
                sentA = self.documents.iloc[index]['sentence_A'].lower().split()
                sentB = self.documents.iloc[index]['sentence_B'].lower().split()
                score = self.cos_sim_lsi(sentA,index)
                file.write(str(score) + "\n")
                print(score)
    def _tfidf(self,save_file):
        with open(save_file,"w") as file:
            Length = len(self.documents)
            for index in range(Length):
                sentA = self.documents.iloc[index]['sentence_A'].lower().split()
                score = self.cos_sim_tfidf(sentA,index)
                file.write(str(score) + "\n")
                print(score)
    def cos_sim_tfidf(self,sentA,index):
        # 对应的句子B Length = len(self.documents)//2
        # indexOfA index --> Length + index
        # 相似度计算的基本方法
        tfidf_model = models.TfidfModel(self.corpus)
        corpus_tfidf = tfidf_model[self.corpus] # 转化
         # 相似度计算的基本方法
        similarity = similarities.MatrixSimilarity(corpus_tfidf)
        answer_corpusA = self.dictionary.doc2bow(sentA)
        answer_tfidfA = tfidf_model[answer_corpusA]
        sims = similarity[answer_tfidfA]
        Length = len(self.documents)
        index = Length + index
        return sims[index]
    def _cosSim(x,y):
        return x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
    def cos_sim_lda(self,sentA,index):
        # corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
        tfidf_model = models.TfidfModel(self.corpus)
        corpus_tfidf = tfidf_model[self.corpus]
        lda_model = models.LdaModel(corpus_tfidf)
        corpus_lda = lda_model[corpus_tfidf]
        similarity_lda = similarities.MatrixSimilarity(corpus_lda)
        # ---------------------------------------------------------------------
        sentA = self.dictionary.doc2bow(sentA)  # 2.转换成bow向量
        answerA_tfidf = tfidf_model[sentA]  # 3.计算tfidf值
        answerA_tfidf = lda_model[answerA_tfidf]  # 4.计算lda值
        result_list = similarity_lda[answerA_tfidf]
        Length = len(self.documents)
        index = Length + index
        return result_list[index]
    def cos_sim_lsi(self,sentA,index):
        # corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
        tfidf_model = models.TfidfModel(self.corpus)
        corpus_tfidf = tfidf_model[self.corpus]
        lsi_model = models.LsiModel(corpus_tfidf)
        corpus_lsi = lsi_model[corpus_tfidf]
        similarity_lsi = similarities.MatrixSimilarity(corpus_lsi)
        # ---------------------------------------------------------------------
        sentA = self.dictionary.doc2bow(sentA)  # 2.转换成bow向量
        answerA_tfidf = tfidf_model[sentA]  # 3.计算tfidf值
        answerA_tfidf = lsi_model[answerA_tfidf]  # 4.计算lda值
        result_list = similarity_lsi[answerA_tfidf]
        Length = len(self.documents)
        index = Length + index
        return result_list[index]
def main():
    test_data = "labeled.txt"
    model = traditional_Model()
    model.load_documents(test_data)
    method = ['tf-idf','lsi','lda']
    for mth in method:
        model.cal_related_ness(mth)
if __name__ == '__main__':
    main()