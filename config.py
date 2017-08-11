#coding=utf-8

import pickle
class Config:

    def __init__(self):
        self.label_dict={} # 标引标记助记符列表
        self.max_len=200 #规范化字串长度
        self.class_label_count=50 # 标记类型总个数
        self.cla_other=self.class_label_count-4 #其它类型索引开始，输出结果不输出的标记类型

