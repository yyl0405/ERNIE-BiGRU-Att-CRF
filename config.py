# coding=utf-8


class Config(object):
    def __init__(self):
        # 微博语料库
        # self.label_file = './data/tag/Weibo_tag.txt'
        # self.train_file = './data/Weibo/weiboNER_train.txt' 
        # self.dev_file   = './data/Weibo/weiboNER_dev.txt'
        # self.test_file  = './data/Weibo/weiboNER_test.txt'
        # eduner语料库
        self.label_file = './data/tag/Eduner_tag.txt'
        self.train_file = './data/eduner/train.txt' 
        self.dev_file   = './data/eduner/dev.txt'
        self.test_file  = './data/eduner/test.txt'
        
        self.vocab = './data/bert/vocab.txt'
        self.max_length = 100
        self.use_cuda = False
        self.gpu = 1
        self.batch_size = 50
        self.ernie_path = './data/bert'
        self.rnn_hidden = 500
        self.ernie_embedding = 768
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 5e-5
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        self.load_model = False
        self.base_epoch = 30
        self.require_improvement = 1000

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])

