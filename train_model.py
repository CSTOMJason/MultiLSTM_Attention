import numpy as np
import tensorflow as tf
import keras
def ReadcleanDatas(cleanpath):
    Datas=None
    with open(cleanpath,"r",encoding="utf-8") as f:
        Datas=f.readlines()[:-1]
    All_Datas=[]
    for line in Datas:
        All_Datas.append(line.split())
    datas=[]
    for per in Datas:
        datas.append(per[2:-1])
    return All_Datas,datas
def GetLabels(All_Datas,labels_onehot):
    targets=[]
    for line in All_Datas:
        targets.append(int(line[0]))
        labels=[]
    for e in targets:
        labels.append(labels_onehot[e])
    return np.array(labels)
    
 #激活函数
def ActivationExpAbs_x(x):
  return tf.exp(-1*tf.abs(x))
 

def attention(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


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


from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
max_feature=50000
tokenizer=Tokenizer(num_words=max_feature)
All_Datas,datas=ReadcleanDatas("./cleandata.txt")
#数据数值化
tokenizer.fit_on_texts(datas)
sequences=tokenizer.texts_to_sequences(datas)
seq_len=32

word2index=tokenizer.word_index
word2index['unk']=0
index2word={index:word for word,index in word2index.items()}
pad_sequences=pad_sequences(sequences,maxlen=seq_len,dtype="int32")
labels_onehot=np_utils.to_categorical(list(NewsDict.values()),num_classes=15,dtype=np.int32)
labels=GetLabels(All_Datas,labels_onehot)
index=np.random.permutation(pad_sequences.shape[0])
pad_sequences=pad_sequences[index]
labels=labels[index]
#print("the pad_sequence.shape is ",pad_sequences.shape)
#print("the labels.shape is ",labels.shape)


#参数设置
import tensorflow as tf
embedding_size=20
hidden_units=32
iterations=50
lr=0.0002
W_fc_keep=tf.placeholder(tf.float32,[],name="W_fc")
with tf.name_scope("XS"):
    xs=tf.placeholder(tf.int32,[None,pad_sequences.shape[1]],name="Input_x")
with tf.name_scope("YS"):
    ys=tf.placeholder(tf.float32,[None,15],name="Input_y")
with tf.name_scope("LSTM_PROB"):
    keep_prob=tf.placeholder(tf.float32,[],name="LSTM_keep_prob")
with tf.name_scope("BATCH_SIZE"):
    batchsize=tf.placeholder(tf.int32,[],name="Batch_size")
with tf.device("/cpu:0"),tf.name_scope("EMBEDDING_LAYER"):
    embedding_matrix=tf.Variable(tf.truncated_normal(shape=[max_feature,embedding_size],mean=0,stddev=0.1),name="Embedding_matrix")
    embedded=tf.nn.embedding_lookup(embedding_matrix,xs,name="Embedded")


with tf.name_scope("Muil_LSTM"):
    lstm_cell_1=tf.nn.rnn_cell.LSTMCell(hidden_units,name="LSTM_CELL_1")
    lstm_cell_2=tf.nn.rnn_cell.LSTMCell(hidden_units,name="LSTM_CELL_2")
    mlstm=tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1,lstm_cell_2])
    drop_mlstm=tf.nn.rnn_cell.DropoutWrapper(mlstm,output_keep_prob=keep_prob)
    init_states=mlstm.zero_state(batchsize,dtype=tf.float32)
    outputs,states=tf.nn.dynamic_rnn(drop_mlstm,embedded,initial_state=init_states,time_major=False)
#添加注意力机制
with tf.name_scope("AttentionLayer"):
    outputs_att=attention(outputs,12)
  
  
#全连接层1
with tf.name_scope("W_FC1_LAYER"):
    W_fc1=tf.Variable(tf.truncated_normal(shape=[32,1024]),name="W_FC1")
    b_fc1=tf.Variable(tf.zeros([1024]),name="b_Fc1")
    out1=tf.matmul(outputs_att,W_fc1)+b_fc1
    out1=ActivationExpAbs_x(tf.nn.dropout(out1,keep_prob=W_fc_keep))

#全连接层2
with tf.name_scope("W_FC2_LAYER"):
    W_fc2=tf.Variable(tf.truncated_normal(shape=[1024,128]),name="W_FC2")
    b_fc2=tf.Variable(tf.zeros([128]),name="b_fc2")
    out2=ActivationExpAbs_x(tf.matmul(out1,W_fc2)+b_fc2)

with tf.name_scope("PRED_LAYER"):
    W_pred=tf.Variable(tf.truncated_normal(shape=[128,15]),name="W_PRED")
    b_pred=tf.Variable(tf.zeros([15]),name="B_pred")
    pred=ActivationExpAbs_x(tf.matmul(out2,W_pred)+b_pred)
 

with tf.name_scope("Cross_Entropy"):
    beta=0.003
    regularization_loss=tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2)+tf.nn.l2_loss(W_pred)
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=ys)+beta*regularization_loss)
tf.summary.scalar("Cross_Entropy",cross_entropy)
with tf.name_scope("Train_op"):
    train_op=tf.train.AdamOptimizer(lr).minimize(cross_entropy)
with tf.name_scope("Accuracy"):
    correct=tf.equal(tf.argmax(pred,1),tf.argmax(ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
tf.summary.scalar("Accuracy",accuracy)
 
 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(pad_sequences,labels,test_size=0.01,random_state=0)
print("x_train.shape is ",x_train.shape )
print("y_train.shape is ",y_train.shape)
print("x_test.shape is ",x_test.shape)

print("y_test.shape is ",y_test.shape ) 




merged=tf.summary.merge_all()
Xcounter=1
saver=tf.train.Saver()

batch_size=1024
num_batch=x_train.shape[0]//batch_size
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer=tf.summary.FileWriter("train",sess.graph)
    for i in range(iterations):
        for step in range(num_batch):
            trax,tray=x_train[step*batch_size:(step+1)*batch_size],y_train[step*batch_size:(step+1)*batch_size]
            _,summary,loss,acc=sess.run([train_op,merged,cross_entropy,accuracy],feed_dict={xs:trax,
                                                                            ys:tray,
                                                                            batchsize:batch_size,
                                                                            keep_prob:0.9,W_fc_keep:0.8})
            train_writer.add_summary(summary,Xcounter)
            Xcounter+=1
            if step%50==0:
                print("%dth In the Traindatas:\n the loss is %f and the accuracy is %f"%(step,loss,acc))
        tempidex=np.random.permutation(1024)
        loss_t,acc_t=sess.run([cross_entropy,accuracy],feed_dict={xs:x_test[tempidex],
                                                                            ys:y_test[tempidex],
                                                                            batchsize:tempidex.shape[0],
                                                                            keep_prob:1.0,
                                                                            W_fc_keep:1.0})
        print("%dth In the Test Datas:\n  the loss is %f and the accuracy is %f"%(step,loss_t,acc_t))
        saver.save(sess,"./model/mlstm_att.ckpt")
        print("the recurrent is %d "%i)
        print("______________________________________________")
            
        
    
