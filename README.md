# MultiLSTM_Attention
今日头条新闻分类
====
数据集链接
----
>>Source数据集(https://drive.google.com/open?id=1do9wGYZzYTT1k4qRx7qlp1cfYdcJyUax)<br/>
>>clean数据集(https://drive.google.com/open?id=1x318dEegr76hhTzRcrkiiyS16TEZ5ZRc)

Requirement
----
    Tensorflow 2.*
    keras
    jieba
    GPU
    sklearn
    re
实验结果导读
----
>>新闻文本分类，使用jieba进行中文分词去掉停用词，采用`多层lstm`网络架构+`Attention layer` (https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)<br/>
>>自定义Activation Function(e-|x|)<br/>
>>>def ActivationFun(x):<br/>
>>>>return tf.exp(-1*tf.abs(x))

数据集清洗
----
    pre_data.py将原始的数据进行去掉标点符号，去掉停止词和jieba分词保存为cleandata.txt
实验超参数
----
    lr=0.0002
    embedding_size=20
    hidden_units=32
    beta=0.003(L2)
    iteration=50
模型结构图
----

  
    
    
        
    
