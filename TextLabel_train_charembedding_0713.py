#coding=utf-8
#keras==2.0.5 tensorflow==1.1.0

import numpy as np
import re
import glob
import os
import codecs
import pickle
# import jieba
from tqdm import tqdm
import config

import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import ModelCheckpoint,Callback
# import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adagrad,Adam
from keras.models import *
from keras.metrics import *
from keras import backend as K
from keras.regularizers import *
# from keras.regularizers import activity_l1 #通过L1正则项，使得输出更加稀疏
from visual_callbacks import AccLossPlotter
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#分词的正则
# split_re=u"[\u2000-\u9fa5]|[a-zA-Z]+|[0-9]|[^\u2000-\u9fa5^a-zA-Z0-9 ]"
split_re = u"[\u2000-\u9fa5]|[a-zA-Z]+|[0-9]+|[ ]|[^\u2000-\u9fa5^a-zA-Z0-9 ]"
#助记符 标引标记
label_dict={
    # 'C_SO':1,'C_SO_M':2,'C_SO_E':3,'C_SO_D':4,
    # 'C_AU1':5,'C_AU1_M':6,'C_AU1_E':7,'C_AU1_D':8,
}
f_tag=codecs.open('tag.txt','r')
lines=f_tag.readlines()
for i_line, line in enumerate(lines):
    line=line.strip()
    label_dict[line]=i_line*4+1
    label_dict[line+'_M'] = i_line*4 + 2
    label_dict[line+'_E'] = i_line*4 + 3
    label_dict[line+'_D'] = i_line*4 + 4
f_tag.close()
label_numer=len(label_dict)
label_dict['OTHER']=label_numer+1
label_dict['OTHER_M']=label_numer+2
label_dict['OTHER_E']=label_numer+3
label_dict['OTHER_D']=label_numer+4
# label_dict['MASK']=label_numer+5
# label_dict['MASK_M']=label_numer+6
# label_dict['MASK_E']=label_numer+7
# label_dict['MASK_D']=label_numer+8
dict= sorted(label_dict.items(), key=lambda d:d[1], reverse = False)
print(dict)

#字串最大长度
max_len = 150
#标引类型总数 0类型=pading字符
class_label_count = len(label_dict)+1 #还有一个0类型
#其它类型值
cla_other = len(label_dict)-3
#保存config.pkl
C = config.Config()
C.label_dict = label_dict
C.max_len = max_len
C.class_label_count = class_label_count
C.cla_other = cla_other
print('max_len:',max_len,'\ncla_other:',cla_other,'\nclass_label_count:',class_label_count)
# generate Config file for test phase
with open('config.pkl', 'wb') as config_f:
    pickle.dump(C,config_f,)
    print('Config has been written to config.pkl, and can be loaded when testing to ensure correct results')

# exit()

def line_space_pro(line):
    pattern = re.compile(u"([\u2000-\u9fa5])( )")
    line = pattern.sub(r'\1', line)
    pattern = re.compile(u"( )([\u2000-\u9fa5])")
    line = pattern.sub(r'\2', line)
    # pattern = re.compile(r'([,:)(-]|~|\])( )')
    # line = pattern.sub(r'\1', line)
    # pattern = re.compile(r'( )([,.:)(-]|~|\[)')
    # line = pattern.sub(r'\2', line)
    # pattern = re.compile(r'([\u2000-\u9fa5])(\.)( )')
    # line = pattern.sub(r'\1\2', line)
    # pattern = re.compile(r'(\d) (\d{3}\()')
    # line = pattern.sub(r'\1\2', line)
    # pattern = re.compile(r'(\d) (\d{3}[年,(])')
    # line = pattern.sub(r'\1\2', line)
    return line

def get_file_list(dir):
    #加载文件列表
    file_list=[]
    # for root,dirs,files in os.walk(r'E:\Formax\217 2016-11-21 SheKeYuan\0607-ZhongHuaYiXueHui\zhonghuayixuehui'):
    #     # print(dirs)
    #     for name in files:
    #         file_list.append(root+'\\'+name)
    #     for dir in dirs:
    #         # print(root+'\\'+dir+'\\*.xml')
    #         for file in glob.glob(root+'\\'+dir+'\\*.xml'):
    #             file_list.append(file)
    #             print(file)
    #     break
    for file in glob.glob(dir):
        file_list.append(file)
    print(len(file_list))
    return file_list

zhujifu_list=['C_DA','C_DT']
zhujifu_del_list=['<REFERENCE_SINGLE>','</REFERENCE_SINGLE>']
# f_tag=codecs.open('mix-tag.txt','r')
# lines=f_tag.readlines()
# for i_line, line in enumerate(lines):
#     line=line.strip()
#     # print(line)
#     zhujifu_del_list.append(line)
# f_tag.close()
#删除/替换ZhuJiFu
def Del_ZhuJiFu(line):
    # print(label_dict.keys())
    # print(line)

    line = line.replace('<C_NU></C_NU>', '')
    line = line.replace('<C_DA></C_DA>', '')
    line = line.replace('<C_DT></C_DT>', '')
    for zhujifu in zhujifu_del_list:
        line = line.replace(zhujifu, '')

    for o in zhujifu_list:
        pattern = re.compile(r'(\<(?P<key>'+o+')\>.*?\</(?P=key)\>)')
        line = pattern.sub(r'', line)

    for d in list(label_dict.keys()):
        line = line.replace('<'+d+'>', '')
        line = line.replace('</' + d + '>', '')
        # line = line.replace('   ','  ')
        # line = line.replace('<sub>', '')
        # line = line.replace('</sub>', '')
        # line = line.replace('<sup>', '')
        # line = line.replace('</sup>', '')
        # line = line.replace('<italic>', '')
        # line = line.replace('</italic>', '')

    line=line.strip()
    # print(line)
    return line
def Replace_ZhuJiFu(line):
    #将训练库里的助记符替换
    # print(label_dict.keys())
    # print(line)
    for zhujifu in zhujifu_del_list:
        line = line.replace(zhujifu, '')

    for o in zhujifu_list:
        pattern = re.compile(r'(\<(?P<key>'+o+')\>.*?\</(?P=key)\>)')
        line = pattern.sub(r'', line)

    # 先从后助记符结束处截断
    for d in list(label_dict.keys()):
        line = line.replace('</' + d + '>', '  ')
    line = line.strip()
    # print(line)
    line_new = ''
    for block in line.split('  '):
        block_new = ''
        b_find = False
        find_index = 0
        for d in list(label_dict.keys()):
            if block.find('<' + d + '>') >= 0:
                find_index = block.find('<' + d + '>')
                if find_index == 0: # 如果助记符在开头
                    block_new = block.replace('<' + d + '>', '')
                    block_new += '=' + str(label_dict[d])
                else:
                    block_new = block.replace('<' + d + '>', '=' + str(cla_other) + '  ')
                    block_new += '=' + str(label_dict[d])
                b_find = True
                break
        # if b_find:
        #     for block_x in block_new.split('  '):
        #
        # print(block_new)
        if not b_find:
            block_new=block+'='+str(cla_other)+'  '
        line_new += block_new + '  '
    line = line_new
    line = line.strip()
    # print(line)
    return line

def existBOM(file_name):
    #utf-8文件是否有BOM头
    file_obj = open(file_name,'rb')
    code = file_obj.read(3)
    file_obj.close()
    if code == codecs.BOM_UTF8:#判断是否包含EF BB BF
        return  True
    return False

#加载word_list.npy 进行转编码
def getXY(file_path):
    #加载字典
    f=open('char_value_dict.pkl','rb')
    import pickle
    char_value_dict=pickle.load(f)
    f.close()
    # print('字典大小:',char_value_dict)

    X=[]
    Y=[]
    f = codecs.open(file_path, 'r', encoding='utf-8' if not existBOM(file_path) else 'utf-8-sig')
    lines = f.readlines()
    f.close()
    print(len(lines))

    # lines=lines[0:100]

    # 处理zhujifu

    # 随机排序
    np.random.seed(1)
    lines_index_list = np.random.randint(0, len(lines), len(lines))
    # if train_test=='train':
    #     lines_index_list = lines_index_list[0:len(lines_index_list)*0.9]
    #     # print('train:',len(lines_index_list),lines_index_list)
    # else:
    #     lines_index_list = lines_index_list[len(lines_index_list)*0.9:]
    #     # print('test:',len(lines_index_list), lines_index_list)


    # for line in lines:
    show_info = False
    for line_index in lines_index_list:
        line = lines[line_index]
        x = []
        y = []

        line = line.strip()
        line = line.replace('  ', ' ')
        line = line_space_pro(line)
        line = Replace_ZhuJiFu(line)
        if show_info:
            print(line)
        for line_split in line.split('  '):
            words = ''
            cla = -1
            if line_split.find('=') >= 0:
                if str(str(line_split).rsplit('=', 1)[1]).isdigit():
                    if int(str(line_split).rsplit('=', 1)[1]) in label_dict.values():
                        # 如果标记的类型
                        words = str(line_split).rsplit('=', 1)[0]
                        cla = int(str(line_split).rsplit('=', 1)[1])
                    else:
                        words = line_split
                        cla = cla_other
                else:
                    words = line_split
                    cla = cla_other
            else:
                # 该块没标记类型
                words = line_split
                cla = cla_other
            # words = words.replace('&lt;', '<')
            # words = words.replace('&gt;', '>')
            # words_split = re.findall(split_re, words)
            # words_split = jieba.cut(words)#分词
            # print(type(words_split))
            # words_split=list(words_split)
            words_split = []
            for c in words:
                # char_vec = ord(c)
                # words_split.append(char_vec)
                # if char_vec > 65535 or char_vec < 0:
                #     print('**********************************************************')
                if c in char_value_dict:
                    words_split.append(char_value_dict[c])
                else:
                    words_split.append(len(char_value_dict) + 1)

            # print(list(words_split))
            # print(words_split,cla)
            for i, word in enumerate(words_split):
                label = cla + 1
                if len(words_split) == 1:
                    if i == 0:
                        label = cla + 3
                else:
                    if i == 0:
                        label = cla
                    if i == len(words_split) - 1:
                        label = cla + 2
                x.append(words_split[i])
                lab = [0] * class_label_count
                lab[label] = 1
                y.append(lab)

        # print(len(x))
        if len(x) > max_len:
            x = x[0:max_len]
            y = y[0:max_len]
        else:
            len_x = len(x)
            for i in range(max_len - len_x):  # 补0
                # x.append(1)
                x.append(0)
                lab = [0] * class_label_count
                lab[0] = 1
                y.append(lab)
        # print(x)
        # if i_line==0:
        #     print(x)
        #     print(np.argmax(y,axis=1))
        #     print(np.sum(np.array(x)>0))

        X.append(x)
        Y.append(y)
    return X,Y
def getXY_gen(file_path, batch_size=32):
    #file_list带Zhujifu的xml文件

    # file_list=get_file_list(r'E:\Formax\217 2016-11-21 SheKeYuan\Program_python\rnn-lstm-char-embedding\ZhongHuaYiXUeHui_program\data\1671-0274-19-09-004.xml')
    #加载字典
    f=open('char_value_dict_0727.pkl','rb')
    import pickle
    char_value_dict=pickle.load(f)
    f.close()
    # print('字典大小:',char_value_dict)

    X=[]
    Y=[]
    while(1):
        f = codecs.open(file_path, 'r', encoding='utf-8' if not existBOM(file_path) else 'utf-8-sig')
        lines=f.readlines()
        f.close()
        print(len(lines))

        # lines=lines[0:100]

        #处理zhujifu

        #随机排序
        np.random.seed(1)
        lines_index_list = np.random.randint(0, len(lines), len(lines))
        # if train_test=='train':
        #     lines_index_list = lines_index_list[0:len(lines_index_list)*0.9]
        #     # print('train:',len(lines_index_list),lines_index_list)
        # else:
        #     lines_index_list = lines_index_list[len(lines_index_list)*0.9:]
        #     # print('test:',len(lines_index_list), lines_index_list)


        # for line in lines:
        show_info=True
        for line_index in lines_index_list:
            line = lines[line_index]
            x = []
            y = []

            line=line.strip()
            line=line.replace('  ',' ')
            # line=line_space_pro(line)
            line=Replace_ZhuJiFu(line)
            if show_info:
                print(line)
            for line_split in line.split('  '):
                words=''
                cla=-1
                if line_split.find('=')>=0:
                    if str(str(line_split).rsplit('=',1)[1]).isdigit():
                        if int(str(line_split).rsplit('=',1)[1]) in label_dict.values():
                            #如果标记的类型
                            words=str(line_split).rsplit('=',1)[0]
                            cla=int(str(line_split).rsplit('=',1)[1])
                        else:
                            words = line_split
                            cla = cla_other
                    else:
                        words = line_split
                        cla = cla_other
                else:
                    #该块没标记类型
                    words=line_split
                    cla=cla_other
                # words = words.replace('&lt;', '<')
                # words = words.replace('&gt;', '>')
                words_split_src = re.findall(split_re, words)
                # words_split = jieba.cut(words)#分词
                # print(type(words_split))
                # words_split=list(words_split)

                words_split=[]
                for word in words_split_src:
                    # char_vec = ord(c)
                    # words_split.append(char_vec)
                    # if char_vec > 65535 or char_vec < 0:
                    #     print('**********************************************************')
                    if word in char_value_dict:
                        words_split.append(char_value_dict[word])
                    else:
                        words_split.append(len(char_value_dict)+1)

                # print(list(words_split))
                # print(words_split,cla)
                for i,word in enumerate(words_split):
                    label = cla+1
                    if len(words_split)==1:
                        if i==0:
                            label=cla+3
                    else:
                        if i==0:
                            label=cla
                        if i==len(words_split)-1:
                            label=cla+2
                    x.append(words_split[i])
                    lab = [0] * class_label_count
                    lab[label] = 1
                    y.append(lab)

            # print(len(x))
            if len(x)>max_len:
                x=x[0:max_len]
                y=y[0:max_len]
            else:
                len_x=len(x)
                for i in range(max_len-len_x):#补0
                    # x.append(1)
                    x.append(0)
                    lab = [0] * class_label_count
                    lab[0] = 1
                    y.append(lab)

            if show_info:
                print(x)
                print(np.argmax(y,axis=1))
            X.append(x)
            Y.append(y)
            show_info = False
            if len(X) == batch_size:
                # print(np.shape(X))
                # print(X)
                x1 = X[0:batch_size]
                y1 = Y[0:batch_size]
                # print(len(x1),len(y1))
                # print(x1)
                # print('*'*100)
                X = []
                Y = []
                yield np.array(x1), np.array(y1)

    # return X,Y

def train():
    # X,Y=getXY()
    # print('X:\n',X)
    # print('Y:\n',Y)
    # print(np.shape(X))
    # print(np.shape(Y))
    # print(X[0])
    # x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1, random_state=1)

    train_file='./data/train.txt'
    test_file='./data/test.txt'
    #加载训练数据
    f = codecs.open(train_file, 'r', encoding='utf-8' if not existBOM(train_file) else 'utf-8-sig')
    lines = f.readlines()
    f.close()
    print(len(lines))
    lines_count_train=len(lines)
    #加载测试数据
    f = codecs.open(test_file, 'r', encoding='utf-8' if not existBOM(test_file) else 'utf-8-sig')
    lines = f.readlines()
    f.close()
    print(len(lines))
    lines_count_test=len(lines)
    #加载字典
    f=open('char_value_dict_0727.pkl','rb')
    import pickle
    char_value_dict=pickle.load(f)
    f.close()
    print('字典大小:',len(char_value_dict))

    #建立模型
    sequence = Input(shape=(max_len,), dtype='int32')
    # mask = Masking(mask_value=0.)(sequence)
    embedded = Embedding(len(char_value_dict)+2, 128, input_length=max_len, mask_zero=True)(sequence)#, trainable=False
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5), merge_mode='sum')(embedded)#dropout_U=0.5, dropout_W=0.5
    blstm = Dropout(0.5)(blstm)
    output = TimeDistributed(Dense(class_label_count, activation='softmax'))(blstm) # activity_regularizer=activity_l1(0.01)
    model = Model(inputs=sequence, outputs=output)

    #加载预训练模型
    # if os.path.exists('bilstm_0727_k205_tf110.hdf5'):
    #     model=load_model('bilstm_0727_k205_tf110.hdf5')
    # model.layers[1].trainable = False#Embedding层不再训练

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #rmsprop
    #mse mae mape msle binary_crossentropy categorical_crossentropy sparse_categorical_crossentropy
    #recall  'categorical_crossentropy'
    #编译模型
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc',])

    # 用于保存验证集误差最小的参数，当验证集误差减少时，立马保存下来
    checkpointer = ModelCheckpoint(filepath="bilstm_0727_k205_tf110.hdf5", verbose=0, save_best_only=True, )
    history = LossHistory()

    # history = model.fit(x_train, y_train,
    #                     batch_size=128, nb_epoch=500,validation_data = (x_test, y_test),
    #                     callbacks=[checkpointer, history, plotter],
    #                     verbose=1
    #                     )

    #训练模型
    model.fit_generator(getXY_gen(train_file, batch_size=32),
                        steps_per_epoch=lines_count_train/32,
                        epochs=300,#*100  1460692 lines_count
                        validation_data=getXY_gen(test_file, batch_size=32),
                        validation_steps=lines_count_test/32,
                        callbacks=[checkpointer, history, plotter]
                        )
def test(file_path):
    #检测模型正确性
    X,Y=getXY(file_path)
    # print(X[1])
    # Y_argmax = np.argmax(Y[1], axis=1)
    # print(Y_argmax)
    #
    # exit()

    # f_src = codecs.open(r'data\fund .src', encoding='utf-8')
    # lines_src = f_src.readlines()
    # f_src.close()
    #
    # f_out = codecs.open(r'data\fund.lst.out', encoding='utf-8')
    # lines_out = f_out.readlines()
    # f_out.close()

    model = load_model('bilstm_0714_k205_tf110.hdf5')
    # print(y_hat)
    error_line_count = 0
    error_chars_count = 0
    chars_count = 0
    for i in (range(len(X))):
        # print(i+1)
        # if i>1000:
        #     break
        y_hat = model.predict(np.array([X[i]]))
        # print(np.argmax(y_hat,axis=2))
        # print(np.shape(Y))
        # print(np.argmax(Y[0],axis=1))
        y_hat_argmax=np.argmax(y_hat,axis=2)
        Y_argmax = np.argmax(Y[i],axis=1)
        y_hat_argmax=np.reshape(y_hat_argmax,(-1))
        Y_argmax=np.reshape(Y_argmax,(-1))
        # print(y_hat_argmax)
        # print(Y_argmax)
        error_char_count=0
        for j in range(len(Y_argmax)):
            chars_count+=1
            if Y_argmax[j]==0:
                break
            if Y_argmax[j]!=y_hat_argmax[j]:
                error_char_count+=1
                error_chars_count+=1
        if error_char_count>0:
            error_line_count+=1
            print(i+1,'********')
            # print(lines_src[i])
            # line=lines_src[i]
            # line = line.strip()
            # line = line.replace('  ', ' ')
            # line = line_space_pro(line)
            # line = Replace_ZhuJiFu(line)
            # print(line)
            print(error_char_count)
            # print(y_hat_argmax)
            # print(Y_argmax)

            # print(lines_src[lines_out])
    print('error_line_count:',error_line_count,error_line_count/len(X))
    print('chars_count:',chars_count)
    print('error_chars_count:',error_chars_count)
    # print('error_char_count:',error_char_count)

#得到 initPro transPro used by viterbi算法
def get_initPro_transPro(file_list):
    # file_list=get_file_list()

    # 标注统计信息
    tagSize = class_label_count
    tagCnt = [0 for i in range(tagSize)]
    tagTranCnt = [[0 for i in range(tagSize)] for j in range(tagSize)]

    for file in file_list:
        f = codecs.open(file, 'r', encoding='utf-8')
        lines=f.readlines()
        f.close()
        # print(lines)

        # lines=lines[0:2000]

        #处理zhujifu
        for i_line,line in (enumerate(lines)):
            print(i_line+1,len(lines))

            line = line.strip()
            line = line.replace('  ', ' ')
            line = line_space_pro(line)
            line = Replace_ZhuJiFu(line)
            tags = []
            # print(line)
            for line_split in line.split('  '):
                words = ''
                cla = -1
                if line_split.find('=') >= 0:
                    if str(str(line_split).rsplit('=', 1)[1]).isdigit():
                        if int(str(line_split).rsplit('=', 1)[1]) in label_dict.values():
                            # 如果标记的类型
                            words = str(line_split).rsplit('=', 1)[0]
                            cla = int(str(line_split).rsplit('=', 1)[1])
                        else:
                            words = line_split
                            cla = cla_other
                    else:
                        words = line_split
                        cla = cla_other
                else:
                    # 该块没标记类型
                    words = line_split
                    cla = cla_other

                # words_split = []
                # for c in words:
                #     char_vec = ord(c)
                #     words_split.append(char_vec)
                    # if char_vec > 65535 or char_vec < 0:
                    #     print('**********************************************************')

                for i, word in enumerate(words):
                    label = cla + 1
                    if len(words) == 1:
                        if i == 0:
                            label = cla + 3
                    else:
                        if i == 0:
                            label = cla
                        if i == len(words) - 1:
                            label = cla + 2
                    tags.append(label)
                    # x.append(words_split[i])
                    # lab = [0] * class_label_count
                    # lab[label] = 1
                    # y.append(lab)

            # print(len(x))
            if len(tags) > max_len:
                pass
                # x = x[0:max_len]
                # y = y[0:max_len]
            else:
                # len_x = len(x)
                for i in range(max_len - len(tags)):  # 补0
                    tags.append(0)
                    # x.append(0)
                    # lab = [0] * class_label_count
                    # lab[0] = 1
                    # y.append(lab)
            #
            # X.append(x)
            # Y.append(y)

            # 统计标注信息
            lineVecY = []
            lastTag = -1
            for tag in tags:
                # 统计tag频次
                tagCnt[tag] += 1
                # 统计tag转移频次
                if lastTag != -1:
                    tagTranCnt[lastTag][tag] += 1
                # 暂存上一次的tag
                lastTag = tag

    # 字总频次
    charCnt = sum(tagCnt)
    # 转移总频次
    tranCnt = sum([sum(tag) for tag in tagTranCnt])
    # tag初始概率
    initProb = []
    for i in range(tagSize):
        initProb.append(tagCnt[i] / float(charCnt))
    # tag转移概率
    tranProb = []
    for i in range(tagSize):
        p = []
        for j in range(tagSize):
            p.append(tagTranCnt[i][j] / float(tranCnt))
        tranProb.append(p)
    output = open('initProb.pkl', 'wb')
    pickle.dump(initProb,output)
    output.close()
    output = open('tranProb.pkl', 'wb')
    pickle.dump(tranProb,output)
    output.close()
    print('*********************')
    print('initProb:',initProb)
    print('*********************')
    print('tranProb:',tranProb)

#从训练数据里得到字典
def get_dict(file_path):
    char_value_dict={}
    f = codecs.open(file_path, 'r', encoding='utf-8' if not existBOM(file_path) else 'utf-8-sig')
    lines = f.readlines()
    f.close()
    print('sentence :',len(lines))
    print('create dictionary...')
    char_count_dict={}
    for i_line, line in tqdm(enumerate(lines)):
        # print(i_line + 1, len(lines))
        line = line.strip()
        line = line.replace('  ', ' ')
        line = Del_ZhuJiFu(line)
        if line.find('>')>=0 or line.find('|')>=0:
            print(line)
        # print(line)
        words_split = re.findall(split_re, line)
        # print('|'.join(words_split))
        for word in words_split:
            if word in char_count_dict:
                char_count_dict[word]+=1
            else:
                char_count_dict[word]=1

    # print(char_count_dict)
    # print(len(char_count_dict))
    dict = sorted(char_count_dict.items(), key=lambda d: d[1], reverse=False)
    # print(dict)
    # print(dict[0][0],dict[0][1])
    start_index=int(0.1*len(dict))
    for i in range(len(dict)):
        print(dict[i])

    value_start=1
    for i in range(start_index,len(dict)):
        char_value_dict[dict[i][0]]=value_start
        value_start+=1
    # print(char_value_dict)
    f=open('char_value_dict_0727.pkl','wb')
    import pickle
    pickle.dump(char_value_dict, f)
    f.close()
    print('write dict ok. len=',len(char_value_dict))


if __name__=='__main__':
    get_dict('./data/train.txt')
    exit()

    # file_list=get_file_list(r'.\data\fund-0705.train')

    # 得到 initPro transPro
    # get_initPro_transPro(file_list)
    train()
    # test('data/train.data')

