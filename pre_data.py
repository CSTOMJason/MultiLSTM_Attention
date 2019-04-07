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

 """#停止词的综合
with open("./stopwords.txt","r",encoding="utf-8") as f:
    stopsw=f.readlines()
stopsw[0]=stopsw[0][1:]
uniquewords=list(set(stopsw))
with open("./stopwords.txt","w",encoding="utf-8") as f:
    for word in uniquewords:
        f.write(word)
       """
 
 def DropPunctuation(sourcepath):
    with open(sourcepath,"r",encoding="utf-8") as f:
        newsdata=f.readlines()
    with open("./nopunctuation.txt","w",encoding="utf-8") as f:
        for line in newsdata:
            f.write(line.split("_!_",3)[2]+" "+re.sub(r"[\W0-9a-zA-Z__丨]","",line.split("_!_",3)[3])+"\n")

def ReadStopwords(path):
    with open(path,"r",encoding="utf-8") as f:
        stopwords=f.readlines()
    for i in range(len(stopwords)):
        stopwords[i]=stopwords[i].strip("\n")
    return stopwords
    
 
 
def WriteCleanData(nodroppath,targetpath):
    stopwords=ReadStopwords("./stopwords.txt")
    with open(nodroppath,"r",encoding="utf-8") as f:#./nopunctuation.txt
        Datas=f.readlines()
    for i in range(len(Datas)):
        Datas[i]=Datas[i].strip("\n")
    NewDatas=[]
    counter=1
    for line in Datas:
        temp=line.split()
        if len(temp)==2:
            target=str(NewsDict.get(temp[0]))
            content=temp[1]
            jiebatemp=list(jieba.cut(content))
            for word in jiebatemp:
                if word not in stopwords:
                    target+=" "+word
            NewDatas.append(target)
            counter+=1
            if counter%500==0:
            print(counter)
    with open(targetpath,"w",encoding="utf-8") as f:#./cleandata.txt
        for line in NewDatas:
            f.write(line+"\n")
if __name__=="__main__":
    WriteCleanData("./nopunctuation.txt","./cleandata.txt")


    
            