import tensorflow as tf
import numpy as np
import re
import jieba
NewsDict={'news_culture':0,
         'news_agriculture':1,
         'news_story':2,
         'news_tech':3,
         'news_finance':4,
         'news_travel':5,
         'news_military':6,
         'news_house':7,
         'news_world':8,
         'news_sports':9,
         'news_edu':10,
         'news_game':11,
         'stock':12,
         'news_car':13,
         'news_entertainment':14}
class MODEL_PREDICTION:
    def __init__(self,text,dictpath,stopwordspath):
        self.dictpath=dictpath
        self.stopwordspath=stopwordspath
        self.text=text
        self.cleantext=self.GetCleanTrue()
       

    def ReadDict(self):
        with open(self.dictpath,"r",encoding="utf-8") as f:
            word2index=f.readlines()
        temp=[]
        for line in word2index:
            temp.append(line.split())
        Dict={w:int(v) for w,v in temp}
        return Dict
    def ReadStopwords(self):
        with open(self.stopwordspath,"r",encoding="utf-8") as f:
            stopwords=f.readlines()
        for i in range(len(stopwords)):
            stopwords[i]=stopwords[i].strip("\n")
        return stopwords

    

    def GetCleanTrue(self):
        word2index=self.ReadDict()
        from keras.preprocessing.sequence import pad_sequences
        temp=re.sub(r"[\W0-9a-zA-Z__丨]","",self.text)
        temp=list(jieba.cut(temp))
        stopwords=self.ReadStopwords()
        for word in temp:
            if word in stopwords:
                temp.remove(word)
        tempindex=[]
        for word in temp:
            if word not in word2index:
                tempindex.append(0)
            else:
                tempindex.append(word2index[word])
        result=pad_sequences(np.array([tempindex]),maxlen=32,dtype="int32")

        return result   


    

def Model_to_pred(textindex):
  result=None
  with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('./model/mlstm_att.ckpt.meta')
    new_saver.restore(sess,"./model/mlstm_att.ckpt")
    graph = tf.get_default_graph()
    prediction=graph.get_tensor_by_name("PRED_LAYER/PRED:0")
    lstm_keep=graph.get_tensor_by_name("LSTM_PROB/LSTM_keep_prob:0")
    wf_keep=graph.get_tensor_by_name("W_fc:0")
    batch_size=graph.get_tensor_by_name("BATCH_SIZE/Batch_size:0")
    xs=graph.get_tensor_by_name("XS/Input_x:0")
#     ys=graph.get_tensor_by_name("YS/Input_y:0")
#     accur=graph.get_tensor_by_name("Accuracy/ACC:0")
 
    feed_dict={xs:textindex,batch_size:1,wf_keep:1.0,lstm_keep:1.0}
    result=sess.run(prediction,feed_dict)
    RNewsDict={index:word for word,index in NewsDict.items()}
    label=np.argmax(result)
    
    return RNewsDict[label]
    
 
if __name__=="__main__":    
    a=MODEL_PREDICTION("今日头条报道，我国的在中小学教育的投资还是存在问题将于下半年整顿！","./word2index.txt","./stopwords.txt")
    result=Model_to_pred(a.cleantext)
    print("这篇新闻是关于--> ",result)
