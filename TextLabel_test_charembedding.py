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
import viterbi
import config

import tensorflow as tf
from keras.callbacks import ModelCheckpoint,Callback
# import keras.backend as K
# from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional,SimpleRNN
# from keras.models import Model
# from keras.optimizers import SGD, RMSprop, Adagrad,Adam
from keras.models import Sequential, model_from_json,load_model
# from visual_callbacks import AccLossPlotter
# plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
# import ZhongKeYuan_train
from keras import backend as K
from keras.preprocessing import sequence


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 分词的正则
# split_re = u"[\u2000-\u9fa5]|[a-zA-Z]+|[0-9]|[ ]|[^\u2000-\u9fa5^a-zA-Z0-9 ]"
split_re = u"[\u2000-\u9fa5]|[a-zA-Z]+|[0-9]+|[ ]|[^\u2000-\u9fa5^a-zA-Z0-9 ]"


#加载配置项config.pkl
with open('config.pkl', 'rb') as f_in:
    C = pickle.load(f_in)
label_dict=C.label_dict # 标引标记
max_len=C.max_len #字串最大长度
class_label_count=C.class_label_count # 总标记类型个数
cla_other=C.cla_other #其它类型开始索引，输出时不输出的标记
print('max_len:',max_len,'\ncla_other:',cla_other,'\nclass_label_count:',class_label_count)
dict= sorted(label_dict.items(), key=lambda d:d[1], reverse = False)
print('label dict:',dict)

label_txt={ v:k for k,v in label_dict.items()}
label_string=[k for k,v in label_dict.items()]
start_index=[i for i in range(1,len(label_dict),4)] #list(label_txt.keys())#开始
# start_index.extend([80,84,88,92,96])
start_index.sort()
mid_index=[i+1 for i in start_index]#结束
end_index=[i+2 for i in start_index]#结束
single_index=[i+3 for i in start_index]#单字符
print('start_index:',start_index)
print('mid_index:',mid_index)
print('end_index:',end_index)

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
    file_list = []
    if dir.find('*.')>=0:
        for file in glob.glob(dir):
            file_list.append(file)
    else:
        for root,dirs,files in os.walk(dir):
            # print(dirs)
            for file in files:
                if file[-4:]=='.ref':
                    file_list.append(os.path.join(root,file))
            # for dir in dirs:
            #     # print(root+'\\'+dir+'\\*.xml')
            #     for file in glob.glob(root+'\\'+dir+'\\*.ref'):
            #         file_list.append(file)
            #         # print(file)
            # break

    print('file count:',len(file_list))
    return file_list

zhujifu_list=[]#['C_LA','C_DA']
zhujifu_del_list=['<fund_text>','</fund_text>']
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
    line=line.replace( '<fund_text>', '')
    line=line.replace( '</fund_text>', '')
    line=line.replace( '<fd>', '')
    line=line.replace( '</fd>', '')
    line=line.replace( '<fund_name>', '')
    line=line.replace( '</fund_name>', '')
    line=line.replace( '<fund_subject>', '')
    line=line.replace( '</fund_subject>', '')
    line=line.replace( '<fund_no>', '')
    line=line.replace( '</fund_no>', '')

    for zhujifu in zhujifu_del_list:
        line = line.replace(zhujifu, '')

    for o in zhujifu_list:
        pattern = re.compile(r'(\<(?P<key>'+o+')\>.+\</(?P=key)\>)')
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
    # print(label_dict.keys())
    # print(line)
    for zhujifu in zhujifu_del_list:
        line = line.replace(zhujifu, '')

    for o in zhujifu_list:
        pattern = re.compile(r'(\<(?P<key>'+o+')\>.+\</(?P=key)\>)')
        line = pattern.sub(r'', line)

    for d in list(label_dict.keys()):
        line = line.replace('<'+d+'>', '  ')
        line = line.replace('</' + d + '>', '='+str(label_dict[d])+'  ')
        line = line.replace('   ','  ')
        # line = line.replace('<sub>', '')
        # line = line.replace('</sub>', '')
        # line = line.replace('<sup>', '')
        # line = line.replace('</sup>', '')
        # line = line.replace('<italic>', '')
        # line = line.replace('</italic>', '')

    line=line.strip()
    return line


def get_line_vec(line,char_value_dict):
    #line文本转固定长度向量

    X=[]
    # Y=[]
    x = []
    # y = []
    w=[]
    line = line.strip()
    line = line.replace('  ', ' ')
    line = line_space_pro(line)

    # words_split = re.findall(split_re, line)
    # words_split = jieba.cut(line)#分词
    # words_split = []
    # for c in line:
    #     # char_vec = ord(c)
    #     # words_split.append(char_vec)
    #     # if char_vec > 65535 or char_vec < 0:
    #     #     print('**********************************************************')
    #     if c in char_value_dict:
    #         words_split.append(char_value_dict[c])
    #     else:
    #         words_split.append(len(char_value_dict) + 1)
    #     w.append(c)
    words_split_src = re.findall(split_re, line)
    # words_split = jieba.cut(words)#分词
    # print(type(words_split))
    # words_split=list(words_split)

    words_split = []
    for word in words_split_src:
        w.append(word)
        # char_vec = ord(c)
        # words_split.append(char_vec)
        # if char_vec > 65535 or char_vec < 0:
        #     print('**********************************************************')
        if word in char_value_dict:
            words_split.append(char_value_dict[word])
        else:
            words_split.append(len(char_value_dict) + 1)

    # print(words_split,cla)
    for i, word in enumerate(words_split):
        x.append(words_split[i])

    # print(len(x))
    if len(x) > max_len:
        x = x[0:max_len]
        # y = y[0:max_len]
    else:
        len_x = len(x)
        for i in range(max_len - len_x):  # 补0
            x.append(0)

    # print(x)
    # print(y)
    X.append(x)
    # Y.append(y)
    return X,w

import sys
def repro_new_path(sent, pre_path, show_info=False):
    #对 new_path 再次规划处理，
    # print(sent)
    # print(pre_path)

    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(pre_path)):
            ss+=sent[i]+'='+str(pre_path[i])+' '
            # print(sent[i],new_path[i])
        print(ss)

    classes=pre_path.copy()
    # 如果是独立字的话=87，优化前后字
    for i, num in enumerate(classes):
        if num == cla_other+3:  # 独立字
            if i > 0:
                if classes[i - 1] in mid_index:
                    classes[i - 1] += 1
            if i < len(classes) - 1:
                if classes[i + 1] in mid_index:
                    classes[i + 1] -= 1
    #如果是其它类型结束，则后一个字符为开始
    for i, num in enumerate(classes):
        if num == cla_other+2:  # 其它类型结束
            # if i > 0:
            #     if classes[i - 1] in mid_index:
            #         classes[i - 1] += 1
            if i < len(classes) - 1:
                if classes[i + 1] in mid_index:
                    classes[i + 1] -= 1
    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(classes)):
            ss += sent[i] + '=' + str(classes[i]) + ' '
            # print(sent[i],new_path[i])
        print(ss)
    # 如果是标点'，'的话，优化前面一个字
    for i, num in enumerate(classes):
        if (sent[i] == ',' or sent[i] == '》' or sent[i] == '译') and (num in start_index or num == cla_other+3):  # 独立字
            if i > 0:
                if classes[i - 1] in mid_index:
                    classes[i - 1] += 1
    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(classes)):
            ss += sent[i] + '=' + str(classes[i]) + ' '
            # print(sent[i],new_path[i])
        print(ss)
    # 如果是标点'，'的话，优化后面一个字
    # ss=''
    # for i in range(len(sent)):
    #     ss+=sent[i]+'='+str(classes[i])+' '
    #     # print(sent[i],new_path[i])
    # print(ss)
    # for i,num in enumerate(classes):
    #     if (sent[i]==','or sent[i]=='》') and num==87:
    #         print('***',sent[i - 1], sent[i], sent[i + 1])
    for i, num in enumerate(classes):
        if (sent[i] == ',' or sent[i] == '》' or sent[i] == '《') and (num in end_index or num == cla_other+3):  # 独立字
            if i < len(classes) - 1:
                if classes[i + 1] in mid_index:
                    classes[i + 1] -= 1
                if classes[i + 1] in end_index:  # 如果
                    classes[i + 1] -= 2
        if i > 0 and i < len(classes) - 1:
            if sent[i] == ',' and classes[i - 1] in end_index:
                if classes[i + 1] in mid_index:
                    classes[i + 1] -= 1
            if sent[i] == '》' and classes[i] in mid_index and classes[i - 1] in end_index:
                classes[i] -= 1
    # if show_info:##################################################
    #     ss='step '+str(sys._getframe().f_lineno)+': '
    #     for i in range(len(sent)):
    #         ss += sent[i] + '-' + str(classes[i]) + ' '
    #         # print(sent[i],new_path[i])
    #     print(ss)
    # for i in range(1, len(classes) - 1):
    #     if sent[i] == ',' and classes[i] in mid_index:
    #         classes[i] = 87
    #         if classes[i - 1] in mid_index:
    #             classes[i - 1] += 1
    #         if classes[i + 1] in mid_index:
    #             classes[i + 1] -= 1
    #         if classes[i + 1] in end_index:
    #             classes[i + 1] -= 2

    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(classes)):
            ss += sent[i] + '=' + str(classes[i]) + ' '
            # print(sent[i],new_path[i])
        print(ss)
    # 如果结束的左右类型相同并且是中间类型，则改为中间
    for i in range(1, len(classes) - 1):
        if classes[i] in end_index \
                and classes[i - 1] in mid_index and classes[i - 1] == classes[i + 1] \
                and classes[i - 1] + 1 == classes[i]:
            classes[i + 1] -= 1
        if sent[i] == '、' and classes[i] in end_index:
            if classes[i - 1] in mid_index and classes[i + 1] in mid_index:
                classes[i] -= 1
        if sent[i] == '~' :
            if classes[i - 1] in end_index:
                classes[i-1] -= 1
            if classes[i - 1] in single_index:
                classes[i - 1] -= 2
    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(classes)):
            ss += sent[i] + '=' + str(classes[i]) + ' '
            # print(sent[i],new_path[i])
        print(ss)
    # 如果结束后跟中间字则将后面的中间字变开始字
    # for i in range(1, len(classes) - 1):
    #     if classes[i] in end_index and classes[i + 1] in mid_index:
    #         classes[i + 1] -= 1
    #如果中文中间的字符出现截断时将字符设置为中间字符
    # for i in range(1, len(classes) - 1):
    #     if classes[i-1] in mid_index and classes[i] in start_index:
    #         # print(sent[i])
    #         if ord(sent[i-1]) in range(0x3400,0x9fff) \
    #                 and ord(sent[i]) in range(0x3400,0x9fff):
    #             # classes[i - 1]-=1
    #             classes[i]+=1
    #     elif classes[i - 1] in end_index and classes[i] in mid_index:
    #         # print(sent[i])
    #         if ord(sent[i - 1]) in range(0x3400, 0x9fff) \
    #                 and ord(sent[i]) in range(0x3400, 0x9fff):
    #             # classes[i - 1]-=1
    #             classes[i- 1] -= 1
    if show_info:
        ss = 'step ' + str(sys._getframe().f_lineno) + ': '
        for i in range(len(classes)):
            ss += sent[i] + '=' + str(classes[i]) + ' '
            # print(sent[i],new_path[i])
        print(ss)
    #如果该字符与两边字符都为中间字符，类型不同时规范 or classes[i-1] in start_index or classes[i+1] in end_index
    # for i in range(1, len(classes) - 1):
    #     if (classes[i-1] in mid_index ) and classes[i] in mid_index  and classes[i-1]==classes[i+1] \
    #             and classes[i - 1] != classes[i]:
    #         classes[i] = classes[i - 1]
            # if abs(classes[i-1] - classes[i])>2:
            #     if classes[i-1] in start_index:
            #         classes[i]=classes[i-1]+1
            #     elif classes[i-1] in mid_index:
            #         classes[i]=classes[i-1]
    if show_info:##################################################
        ss = 'step ' + str(sys._getframe().f_lineno) + ': '
        for i in range(len(classes)):
            ss += sent[i] + '=' + str(classes[i]) + ' '
            # print(sent[i],new_path[i])
        print(ss)

    # ss=''
    # for i in range(len(sent)):
    #     ss+=sent[i]+'='+str(classes[i])+' '
    #     # print(sent[i],new_path[i])
    # print(ss)
    #
    # return classes

    checksmo = []  # 记录start middle end single
    for i in classes:
        if i in start_index:
            checksmo.append('s')
        elif i in mid_index:
            checksmo.append('m')
        elif i in end_index:
            checksmo.append('e')
        elif i in single_index:
            checksmo.append('o')
        else:
            checksmo.append('o')
    # print(checksmo)
    notchangesmo = [0] * len(checksmo)  # 不需要改变的记录下来，o类型 和正常的s-e类型
    for i, smeo in enumerate(checksmo):
        if smeo == 'o':
            notchangesmo[i] = 1
            continue
        if notchangesmo[i] == 1:
            continue

        if smeo == 's':
            check_ok = 1
            for j in range(i + 1, len(checksmo)):

                if abs(classes[i] - classes[j]) <= 2:
                    check_ok = 1
                else:
                    check_ok = 0
                    break
                if checksmo[j] == 'e':
                    break
            if check_ok == 1:
                notchangesmo[i] = 1
                for j in range(i + 1, len(checksmo)):
                    notchangesmo[j] = 1
                    if checksmo[j] == 'e':
                        break

    # ss=''
    # for i in range(len(sent)):
    #     ss+=sent[i]+'='+str(pre_path[i])+' '
    #     # print(sent[i],new_path[i])
    # print(ss)
    # print(notchangesmo)
    # print(checksmo)

    new_path=classes.copy()

    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(classes)):
            ss += sent[i] + '=' + str(classes[i]) + ' '
            # print(sent[i],new_path[i])
        print(ss)

    #1：人名处理 把'·'变成人名
    # for i in range(1,len(new_path)-1):
    #     if sent[i]=='·':
    #         if (new_path[i-1]>=4 and new_path[i-1]<=15) or (new_path[i+1]>=4 and new_path[i+1]<=15):
    #             new_path[i] = 4
    #             # notchangesmo[i]=1
    #             for j in range(i+1,len(new_path)):
    #                 if new_path[j]>=4 and new_path[j]<=16:
    #                     new_path[j] = 4
    #                     # notchangesmo[j] = 1
    #                 else:
    #                     break

    # print(notchangesmo)
    # print(checksmo)
    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(new_path)):
            ss += sent[i] + '=' + str(new_path[i]) + ' '
            # print(sent[i],new_path[i])
        print(ss)

    #2：不同类型时按占比计算
    for i in range(len(new_path)):
        if len(new_path) <= 0:
            break
        if len(new_path) == 1:
            new_path[i] = cla_other+3
        if notchangesmo[i] == 1:
            # 如果已经符合要求，则不处理
            continue
        if i == 0:
            continue
        if checksmo[i] == 's':
            # 向后找 'e'，如果找到了e则整段标为1
            check_list=[]
            new_path_index=[]
            for j in range(i + 1, len(new_path)):
                if notchangesmo[j] == 1:
                    break
                # if checksmo[j] == 's':
                #     break
                if checksmo[j] == 'e':
                    for x in range(i, j + 1):
                        notchangesmo[x] = 1
                        # check_list.append(new_path[x])
                        new_path_index.append(x)
                        if new_path[x] >=cla_other and new_path[x] <=cla_other+2:
                            continue
                        if new_path[x] in mid_index:
                            check_list.append(new_path[x]-1)
                        elif  new_path[x] in end_index:
                            check_list.append(new_path[x]-2)
                        else:
                            check_list.append(new_path[x])
                        # new_path[x] = new_path[i] + 1
                    # new_path[j] = new_path[i] + 2
                    break
            # print(check_list)
            if len(check_list)>0:
                count_times = []
                for item in check_list:
                    count_times.append(check_list.count(item))
                # print(check_list)
                # print(count_times)
                m = max(count_times)
                n = count_times.index(m)
                # print(check_list[n])
                for item in range(len(new_path_index)):
                    if item==0:
                        new_path[new_path_index[item]] = check_list[n]
                        notchangesmo[new_path_index[item]] = 1
                    elif item==len(new_path_index)-1:
                        new_path[new_path_index[item]] = check_list[n]+2
                        notchangesmo[new_path_index[item]] = 1
                    else:
                        new_path[new_path_index[item]] = check_list[n]+1
                        notchangesmo[new_path_index[item]] = 1
    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(new_path)):
            ss += sent[i] + '=' + str(new_path[i]) + ' '
            # print(sent[i],new_path[i])
        print(ss)

    #对人名进行规范化。1：人名处理
    # for i in range(len(new_path)):
    #     if new_path[i]>=4 and new_path[i]<=15:
    #         new_path[i] = 4
    # for i in range(1, len(new_path) - 1):
    #     if sent[i] == '·':
    #         if (new_path[i - 1] >= 4 and new_path[i - 1] <= 15) or (
    #                 new_path[i + 1] >= 4 and new_path[i + 1] <= 15):
    #             new_path[i] = 4
    #             notchangesmo[i]=1
    #             for j in range(i + 1, len(new_path)):
    #                 if new_path[j] >= 4 and new_path[j] <= 15:
    #                     new_path[j] = 4
    #                     notchangesmo[j] = 1
    #                 else:
    #                     break

                                # for j in range(i+1,len(new_path)):
            #     if new_path[j]<4 or new_path[j]>15:
            #         for item in range(i,j+1):
            #             if item == i:
            #                 new_path[item] = 4
            #             elif item == j+1 - 1:
            #                 new_path[item] = 6
            #             else:
            #                 new_path[item] = 5
            #         break
    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(new_path)):
            ss+=sent[i]+'='+str(new_path[i])+' '
            # print(sent[i],new_path[i])
        print(ss)

    # for i in range(1,len(new_path)-1):
    #     if notchangesmo[i-1]==1 and notchangesmo[i]==0:
    #         if new_path[i] in mid_index:
    #             new_path[i]-=1
    #         for j in range(i,len(new_path)):
    #             if notchangesmo[j]==0 and new_path[j] not in mid_index:
    #                 notchangesmo[j] =1
    #             else:
    #                 break

    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(notchangesmo)):
            ss+=sent[i]+'='+str(notchangesmo[i])+' '
            # print(sent[i],new_path[i])
        print(ss)

    for i in range(len(new_path)):
        if notchangesmo[i] == 0:
            new_path[i] = cla_other+3
    # print(notchangesmo)

    #对年份长度的判断
    # for i in range(len(new_path)):
    #     if new_path[i]==20:
    #         year_char_count=1
    #         for j in range(i+1,len(new_path)):
    #             year_char_count += 1
    #             if new_path[j] in end_index: #new_path[j]>23 or new_path[j]<20:
    #                 break
    #         # print(year_char_count)
    #         if year_char_count!=4:
    #             for j in range(i , len(new_path)):
    #                 new_path[j]=87
    #                 if new_path[j] in end_index: #new_path[j] > 23 or new_path[j] < 20:
    #                     break
    #         else:
    #             for j in range(i+1 , len(new_path)):
    #                 if new_path[j] in end_index: #new_path[j] > 23 or new_path[j] < 20:
    #                     new_path[j] = 22
    #                     break
    #                 new_path[j]=21


    if show_info:##################################################
        ss='step '+str(sys._getframe().f_lineno)+': '
        for i in range(len(new_path)):
           ss+=sent[i]+'='+str(new_path[i])+' '
           #print(sent[i],new_path[i])
        print(ss)

    return new_path

def existBOM(file_name):
    #utf-8文件是否有BOM头
    file_obj = open(file_name,'rb')
    code = file_obj.read(3)
    file_obj.close()
    if code == codecs.BOM_UTF8:#判断是否包含EF BB BF
        return  True
    return False

def get_dict_file():
    #加载字典文件
    f=open('char_value_dict_0727.pkl','rb')
    import pickle
    char_value_dict=pickle.load(f)
    f.close()
    print('字典大小:', len(char_value_dict))
    return char_value_dict

index_label = {0: 'C', 1: 'D', 2: 'J', 3: 'M', 4: 'N', 5: 'P', 6: 'R', 7: 'Z', 8: 'O'}
def get_sent_class(sent, model_class):
    max_len = 150
    X = []
    x = []
    for k, s in enumerate(sent):
        x.append(ord(s))
    # lab = [0] * class_label_count
    # lab[file_class - 1] = 1
    # y.append(lab)

    x1 = sequence.pad_sequences([x], maxlen=max_len, padding='pre')[0]
    # print(x1)
    X.append(list(x1))
    try:
        y_out = model_class.predict(np.array([X[0][0:max_len]]))
        # print(y_out)
        y_pre = np.argmax(np.array(y_out), axis=1)
        # print(str(index_label[y_pre[0]]), y_out[0][y_pre[0]])
        return str(index_label[y_pre[0]]), y_out[0][y_pre[0]]
    except:
        return '',0
def get_sent_template(ss):
    m = re.findall(r"\<(C_\w+)\>", ss)
    template_str=''
    if m:
        # print('|'.join(m))
        template_str+='|'.join(m)
    return template_str

def test_to_out(file_list):
    #生成标引后文件
    with K.get_session():
        char_value_dict = get_dict_file()

        model = load_model('bilstm_0727_k205_tf110.hdf5')
        # model_class = load_model('bilstm_text_classify_0717.hdf5')

        # file_list=[r'.\data\原始数据-中国基层医药2017年7月第24卷第13期.txt']

        for file in file_list:
            print(file)
            if os.path.exists(file+'.out'):
                continue

            f = codecs.open(file, 'r', encoding='utf-8' if not existBOM(file) else 'utf-8-sig')#-sig
            lines = f.readlines()
            f.close()
            # print(lines)

            test_line=False
            # lines=['本文为国家自然科学基金项目“税收缺口测算体系与税收政策分析拓展模型研究(71163044)”的阶段性成果.']

            if not test_line:
                if os.path.exists(file + '.out'):
                    os.remove(file + '.out')

            for i_line,line in (enumerate(lines)):
                # print(i_line+1,len(lines))
                if line.find('<—外文—>') >= 0 or line.find('空参考文献') >= 0 or line.find('<p>') < 0:
                    #保存结果文件
                    f_out=codecs.open(file+'.out',mode='a',encoding='utf-8')
                    f_out.seek(2)
                    f_out.write(line.strip()+'\r\n')
                    f_out.close()
                    continue
                if line.find('<p>') >= 0:
                    line = line.replace('<p>', '')
                    line = line.replace('</p>', '')
                    line=Del_ZhuJiFu(line)
                    line=line_space_pro(line)
                    vec,sent=get_line_vec(line, char_value_dict)
                    vec = np.array(vec)
                    # if test_line :
                    #     print(line)
                    # print('vec:',list(vec[0]))

                    y_out = model.predict(vec)
                    # print(np.argmax(y_out,axis=2))

                    # break
                    # #################################################
                    #使用viterbi算法后处理
                    # y_out[:, len(sent):, :] = 0
                    # y_out[:, len(sent):, 0] = 1
                    # # print('y_out_proba:',y_out)
                    # # print(list(y_out[0][0]))
                    # # print(np.argmax(y_out[0][0]))
                    # # print(np.shape(y_out_proba))
                    # # print(y_out_proba.transpose())
                    # # print(np.shape(y_out_proba.transpose()))
                    # # print(list(np.argmax(y_out, axis=2)[0]))
                    # prob, path = viterbi.viterbi(min(len(vec[0]), len(sent)), class_label_count, initProb, tranProb,
                    #                              y_out.transpose())
                    # # print('prob:',prob)
                    # # print('path:',path)
                    # # if len(y_out[0])>len(path):
                    # #     for i in range(len(y_out[0])-len(path)):
                    # #         path.append(0)
                    # # break
                    # classes = path  # list(np.argmax(y_out[0], axis=1)) #path #
                    # # print(classes)
                    # classes = classes[0:len(sent)]
                    # # print('classes:',classes)

                    #################################################
                    #使用传统算法后处理
                    classes = list(np.argmax(y_out[0], axis=1))
                    # print(classes)
                    classes = classes[0:len(sent)]
                    # print(classes)

                    show_info = False
                    # if test_line:
                    #     show_info=True
                    # 对 new_path 再次规划处理，1：人名处理 2：不同类型时按占比计算 3：C_SO 与 C_TI 的处理
                    new_path = classes.copy()
                    new_path = repro_new_path(sent, classes, show_info=show_info)

                    #输出结果串
                    ss = ''
                    ss_temp = ''
                    teg_pre_char = ''
                    i_cur = 0
                    # print(label_string)
                    for i in range(len(new_path)):
                        # print(label_txt[new_path[i]])
                        teg_pre_char = label_txt[new_path[i]]
                        # if label_txt[new_path[i]].find('_') >= 0:
                        #     teg_pre_char = label_txt[new_path[i]].split('_')[0]
                        if label_txt[new_path[i]][-2:] == '_M':
                            teg_pre_char = label_txt[new_path[i]][:-2]
                        if label_txt[new_path[i]][-2:] == '_E':
                            teg_pre_char = label_txt[new_path[i]][:-2]

                        # print(teg_pre_char)
                        ss_temp = sent[i]
                        # print(i)
                        if i <= i_cur and i > 0:
                            continue
                        for j in range(i + 1, len(new_path)):
                            teg_cur_char = label_txt[new_path[j]]
                            # if label_txt[new_path[j]].find('_') >= 0:
                            #     teg_cur_char = label_txt[new_path[j]].split('_')[0]
                            if label_txt[new_path[j]][-2:] == '_M':
                                teg_cur_char = label_txt[new_path[j]][:-2]
                            if label_txt[new_path[j]][-2:] == '_E':
                                teg_cur_char = label_txt[new_path[j]][:-2]

                            if teg_pre_char == teg_cur_char:
                                ss_temp += sent[j]
                            else:
                                break
                            i_cur = j
                        # print('i=',i)
                        # print(int(new_path[i]))
                        if int(new_path[i]) < cla_other:
                            if label_txt[new_path[i]][-2:] == '_D' and new_path[i] - 3 >= 1:
                                ss += '<' + label_txt[new_path[i] - 3] + '>' + ss_temp + '</' + label_txt[
                                    new_path[i] - 3] + '>'
                            else:
                                ss += '<' + label_txt[new_path[i]] + '>' + ss_temp + '</' + label_txt[new_path[i]] + '>'
                        else:
                            ss += ss_temp
                            # if i_cur == len(new_path) - 1:
                            #     break
                    if len(sent) > max_len: # 如果原始串长度>max_len
                        ss += ''.join(sent[max_len:])
                        #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    # print(ss)
                    if test_line:
                        print(ss)
                        break
                    #################################################
                    # 类型标引
                    class_str, prob = get_sent_class(line, model_class=model_class)
                    # 得到该条记录的Template
                    template_str = get_sent_template(ss)
                    #######################################################
                    #保存结果文件
                    f_out=codecs.open(file+'.out',mode='a',encoding='utf-8')
                    f_out.seek(2)
                    f_out.write('\t\t<template>'+template_str+'</template><class>'+class_str+' '+str(prob)+'</class>'+ss + '\r\n')
                    f_out.close()

                # if i_line>1000:
                #     break
def test_fmt_to_out(file_list):
    #生成标引后文件
    # pkl_file = open('initProb.pkl', 'rb')
    # initProb = pickle.load(pkl_file)
    # pkl_file.close()
    # pkl_file = open('tranProb.pkl', 'rb')
    # tranProb = pickle.load(pkl_file)
    # pkl_file.close()

    with K.get_session():
        char_value_dict = get_dict_file()

        model = load_model('bilstm_0713_k205_tf110.hdf5')

        # file_list=get_file_list(r'E:\Formax\217 2016-11-21 SheKeYuan\Program_python\rnn-lstm-char-embedding\Foundation_Program_0628\data\grade3-0629\grade3')
        # file_list=[r'.\data\原始数据-中国基层医药2017年7月第24卷第13期.txt']

        for file in tqdm(file_list):
            # print(file)
            f = codecs.open(file, 'r', encoding='utf-8-sig')#-sig
            lines = f.readlines()
            f.close()
            # print(lines)

            test_line=False
            # lines=['本文为国家自然科学基金项目“税收缺口测算体系与税收政策分析拓展模型研究(71163044)”的阶段性成果.']

            if not test_line:
                if os.path.exists(file + '.out'):
                    os.remove(file + '.out')
            ss_out_file=''
            for i_line,line in (enumerate(lines)):
                # print(line)
                if line.find('<fd>')<0:
                    ss_out_file+=line
                    # f_out = codecs.open(file + '.out', mode='a', encoding='utf-8')
                    # f_out.seek(2)
                    # f_out.write(line)
                    # f_out.close()
                    continue
                # print(i_line+1,len(lines))
                line=Del_ZhuJiFu(line)
                line=line_space_pro(line)
                vec,sent=get_line_vec(line, char_value_dict)
                vec = np.array(vec)
                # if test_line :
                # print(line)
                # print('vec:',list(vec[0]))

                y_out = model.predict(vec)
                # print(np.argmax(y_out,axis=2))

                # viterbi算法
                # #################################################
                # y_out[:, len(sent):, :] = 0
                # y_out[:, len(sent):, 0] = 1
                # prob, path = viterbi.viterbi(min(len(vec[0]), len(sent)), class_label_count, initProb, tranProb,
                #                              y_out.transpose())
                # # print('prob:',prob)
                # # print('path:',path)
                # # if len(y_out[0])>len(path):
                # #     for i in range(len(y_out[0])-len(path)):
                # #         path.append(0)
                # # break
                # classes = path  # list(np.argmax(y_out[0], axis=1)) #path #
                # # print(classes)
                # classes = classes[0:len(sent)]
                # # print('classes:',classes)

                #################################################
                classes = list(np.argmax(y_out[0], axis=1))
                classes = classes[0:len(sent)]
                # print(classes)

                show_info = False
                # if test_line:
                #     show_info=True
                # 对 new_path 再次规划处理，1：人名处理 2：不同类型时按占比计算 3：C_SO 与 C_TI 的处理
                new_path = classes.copy()
                new_path = repro_new_path(sent, classes, show_info=show_info)

                ss = ''
                ss_temp = ''
                teg_pre_char = ''
                i_cur = 0
                # print(label_string)
                for i in range(len(new_path)):
                    # print(label_txt[new_path[i]])
                    teg_pre_char = label_txt[new_path[i]]
                    # if label_txt[new_path[i]].find('_') >= 0:
                    #     teg_pre_char = label_txt[new_path[i]].split('_')[0]
                    if label_txt[new_path[i]][-2:] == '_M':
                        teg_pre_char = label_txt[new_path[i]][:-2]
                    if label_txt[new_path[i]][-2:] == '_E':
                        teg_pre_char = label_txt[new_path[i]][:-2]

                    # print(teg_pre_char)
                    ss_temp = sent[i]
                    # print(i)
                    if i <= i_cur and i > 0:
                        continue
                    for j in range(i + 1, len(new_path)):
                        teg_cur_char = label_txt[new_path[j]]
                        # if label_txt[new_path[j]].find('_') >= 0:
                        #     teg_cur_char = label_txt[new_path[j]].split('_')[0]
                        if label_txt[new_path[j]][-2:] == '_M':
                            teg_cur_char = label_txt[new_path[j]][:-2]
                        if label_txt[new_path[j]][-2:] == '_E':
                            teg_cur_char = label_txt[new_path[j]][:-2]

                        if teg_pre_char == teg_cur_char:
                            ss_temp += sent[j]
                        else:
                            break
                        i_cur = j
                    # print('i=',i)
                    # print(int(new_path[i]))
                    if int(new_path[i]) < cla_other:
                        if label_txt[new_path[i]][-2:] == '_D' and new_path[i] - 3 >= 1:
                            ss += '<' + label_txt[new_path[i] - 3] + '>' + ss_temp + '</' + label_txt[
                                new_path[i] - 3] + '>'
                        else:
                            ss += '<' + label_txt[new_path[i]] + '>' + ss_temp + '</' + label_txt[new_path[i]] + '>'
                    else:
                        ss += ss_temp
                        # if i_cur == len(new_path) - 1:
                        #     break
                if len(sent) > max_len:
                    ss += ''.join(sent[max_len:])
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                # print(ss)
                if test_line:
                    print(ss)
                    break
                #################################################
                ss_out_file += '<fd>'+ss+'</fd>\r\n'

                # f_out=codecs.open(file+'.out',mode='a',encoding='utf-8')
                # f_out.seek(2)
                # f_out.write('<fd>'+ss+'</fd>\r\n')
                # f_out.close()
            f_out=codecs.open(file+'.out',mode='w',encoding='utf-8')
            f_out.write(ss_out_file)
            f_out.close()

                # if i_line>1000:
                #     break
def test_one_string(line):

    char_value_dict=get_dict_file()

    model = load_model('bilstm_0727_k205_tf110.hdf5')
    # pkl_file = open('initProb.pkl', 'rb')
    # initProb = pickle.load(pkl_file)
    # pkl_file.close()
    # pkl_file = open('tranProb.pkl', 'rb')
    # tranProb = pickle.load(pkl_file)
    # pkl_file.close()

    print(line)
    # line = line.replace('&lt;', '<')
    # line = line.replace('&gt;', '>')

    # print(i_line+1,len(lines))
    line = Del_ZhuJiFu(line)
    # line = line_space_pro(line)
    vec, sent = get_line_vec(line, char_value_dict)
    vec = np.array(vec)
    print(vec)
    # if test_line :
    print(line)
    # print('vec:',list(vec[0]))

    y_out = model.predict(vec)
    # print('y_out:',np.argmax(y_out,axis=2))

    # viterbi算法
    # #################################################
    # y_out[:, len(sent):, :] = 0
    # y_out[:, len(sent):, 0] = 1
    # prob, path = viterbi.viterbi(min(len(vec[0]), len(sent)), class_label_count, initProb, tranProb,
    #                              y_out.transpose())
    # # print('prob:',prob)
    # # print('path:',path)
    # # if len(y_out[0])>len(path):
    # #     for i in range(len(y_out[0])-len(path)):
    # #         path.append(0)
    # # break
    # classes = path  # list(np.argmax(y_out[0], axis=1)) #path #
    # # print(classes)
    # classes = classes[0:len(sent)]
    # # print('classes:',classes)

    #################################################
    classes = list(np.argmax(y_out[0], axis=1))
    classes = classes[0:len(sent)]
    # print(classes)

    show_info = False
    # if test_line:
    #     show_info=True
    # 对 new_path 再次规划处理，1：人名处理 2：不同类型时按占比计算 3：C_SO 与 C_TI 的处理
    new_path = classes.copy()
    new_path = repro_new_path(sent, classes, show_info=show_info)
    print('new_path:', classes)

    ss = ''
    ss_temp = ''
    teg_pre_char = ''
    i_cur = 0
    # print(label_string)
    for i in range(len(new_path)):
        # print(label_txt[new_path[i]])
        teg_pre_char = label_txt[new_path[i]]
        # if label_txt[new_path[i]].find('_') >= 0:
        #     teg_pre_char = label_txt[new_path[i]].split('_')[0]
        if label_txt[new_path[i]][-2:] == '_M':
            teg_pre_char = label_txt[new_path[i]][:-2]
        if label_txt[new_path[i]][-2:] == '_E':
            teg_pre_char = label_txt[new_path[i]][:-2]

        # print(teg_pre_char)
        ss_temp = sent[i]
        # print(i)
        if i <= i_cur and i > 0:
            continue
        for j in range(i + 1, len(new_path)):
            teg_cur_char = label_txt[new_path[j]]
            # if label_txt[new_path[j]].find('_') >= 0:
            #     teg_cur_char = label_txt[new_path[j]].split('_')[0]
            if label_txt[new_path[j]][-2:] == '_M':
                teg_cur_char = label_txt[new_path[j]][:-2]
            if label_txt[new_path[j]][-2:] == '_E':
                teg_cur_char = label_txt[new_path[j]][:-2]

            if teg_pre_char == teg_cur_char:
                ss_temp += sent[j]
            else:
                break
            i_cur = j
        # print('i=',i)
        # print(int(new_path[i]))
        if int(new_path[i]) < cla_other:
            if label_txt[new_path[i]][-2:]=='_D' and new_path[i]-3>=1:
                ss += '<' + label_txt[new_path[i]-3] + '>' + ss_temp + '</' + label_txt[new_path[i]-3] + '>'
            else:
                ss += '<' + label_txt[new_path[i]] + '>' + ss_temp + '</' + label_txt[new_path[i]] + '>'
        else:
            ss += ss_temp
            # if i_cur == len(new_path) - 1:
            #     break
    if len(sent) > max_len:
        ss += ''.join(sent[max_len:])
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(ss)


if __name__=='__main__':
    # file_list=get_file_list(r'.\data\fund.train')
    # test_to_out(file_list)
    # file_list=get_file_list(r'data\grade3-0629\grade3')
    # test_fmt_to_out(file_list)
    # test_one_string('[6] 陶渊明集[M].王瑶, 编注.北京: 人民文学出版社, 1957.')
    # exit()
    # file_list = get_file_list(r'E:\Formax\217 2016-11-21 SheKeYuan\Program_python\rnn-lstm-char-embedding\SheKeYuan_0713\data\grade2-0715\grade2')

    # test_to_out(file_list)

    # test_one_string('<C_NU>1</C_NU> <C_SO>Making ethyne: rsc.li/EiC506p123</C_SO>')
    # test_one_string('(1) Bloom, L. B. (2009) Loading clamps for DNA replication and repair. DNA Repair 8, 570-578.')
    # test_one_string('(2) Goodman, M. F., and Woodgate, R. (2013) Translesion DNA polymerases. Cold Spring Harbor Perspect. Biol. 5, a010363.')
    # test_one_string('(3) Tinker, R. L., Kassavetis, G. A., and Geiduschek, E. P. (1994) Detecting the ability of viral, bacterial and eukaryotic replication proteins to track along DNA. EMBO J. 13, 5330-5337.')
    # test_one_string('(4) Kochaniak, A. B., Habuchi, S., Loparo, J. J., Chang, D. J., Cimprich, K. A., Walter, J. C., and van Oijen, A. M. (2009) Proliferating cell nuclear antigen uses two distinct modes to move along DNA. J. Biol. Chem. 284, 17700-17710.')
    # test_one_string('(5) Hedglin, M., and Benkovic, S. J. (2017) Replication protein A prohibits diffusion of the PCNA sliding clamp along single-stranded DNA. Biochemistry, DOI: 10.1021/acs.biochem.6b01213.')
    test_one_string('(6) Chen, R., and Wold, M. S. (2014) Replication protein A: singlestranded DNA’s first responder: dynamic DNA-interactions allow replication protein A to direct single-strand DNA intermediates into different pathways for synthesis or repair. BioEssays 36, 1156-1161.')
    # test_one_string('(7) Laurence, T. A., Kwon, Y., Johnson, A., Hollars, C. W., O’Donnell, M., Camarero, J. A., and Barsky, D. (2008) Motion of a DNA sliding clamp observed by single molecule fluorescence spectroscopy. J. Biol. Chem. 283, 22895-22906.')


