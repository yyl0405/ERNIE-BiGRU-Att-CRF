# coding=utf-8
import torch.nn as nn
import torch
from transformers import AutoModel
from torchcrf import CRF

# class ERNIE_LSTM_CRF(nn.Module):
#     """
#     ernie_lstm_crf model
#     """
#     def __init__(self, ernie_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False):
#         super(ERNIE_LSTM_CRF, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         #加载ERNIE
#         self.word_embeds = AutoModel.from_pretrained(ernie_config)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim,
#                             num_layers=rnn_layers, bidirectional=True,
#                             dropout=dropout_ratio, batch_first=True)
#         self.rnn_layers = rnn_layers
#         self.dropout1 = nn.Dropout(p=dropout1)
#         self.crf = CRF(num_tags=tagset_size, batch_first=True)
#         self.liner = nn.Linear(hidden_dim*2, tagset_size)
#         self.tagset_size = tagset_size

#     def rand_init_hidden(self, batch_size):
#         """
#         random initialize hidden variable
#         """
#         return Variable(torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), \
#                Variable(torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))


#     def forward(self, sentence, attention_mask=None):
#         '''
#         args:
#             sentence (batch_size, word_seq_len) : word-level representation of sentence
#             hidden: initial hidden state

#         return:
#             crf input (batch_size, word_seq_len, tag_size), hidden
#         '''
#         batch_size = sentence.size(0)
#         seq_length = sentence.size(1)
#         embeds = self.word_embeds(sentence, attention_mask=attention_mask)
#         hidden = self.rand_init_hidden(batch_size)
#         if embeds[0].is_cuda:
#             hidden = tuple(i.cuda() for i in hidden)
#         lstm_out, hidden = self.lstm(embeds[0], hidden)
#         lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)
#         d_lstm_out = self.dropout1(lstm_out)
#         l_out = self.liner(d_lstm_out)
#         lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
#         return lstm_feats

#     def loss(self, feats, mask, tags):
#         """
#         feats: size=(batch_size, seq_len, tag_size)
#             mask: size=(batch_size, seq_len)
#             tags: size=(batch_size, seq_len)
#         :return:
#         """
#         loss_value = -self.crf(feats, tags, mask) # 计算损失
#         batch_size = feats.size(0)
#         loss_value /= float(batch_size)
#         return loss_value


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        Q = self.query(x)  # Query
        K = self.key(x)    # Key
        V = self.value(x)  # Value

        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 使用注意力权重加权求和
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

class ERNIE_GRU_CRF(nn.Module):
    """
    ernie_gru_crf model with self-attention
    """
    def __init__(self, ernie_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False):
        super(ERNIE_GRU_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # 加载ERNIE
        self.word_embeds = AutoModel.from_pretrained(ernie_config)
        # 使用GRU替代LSTM，并设置为双向
        self.gru = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=rnn_layers, bidirectional=True,
                          dropout=dropout_ratio, batch_first=True)
        self.attention = SelfAttention(hidden_dim * 2)  # GRU的输出是hidden_dim * 2
        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(num_tags=tagset_size, batch_first=True)
        self.liner = nn.Linear(hidden_dim * 2, tagset_size)
        self.tagset_size = tagset_size

    def rand_init_hidden(self, batch_size):
        """
        随机初始化隐藏状态
        """
        # 对于GRU，我们只需要初始化一个隐藏状态的张量
        return torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)

    def forward(self, sentence, attention_mask=None):
        '''
        args:
            sentence (batch_size, word_seq_len) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf input (batch_size, word_seq_len, tag_size), hidden
        '''
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        embeds = self.word_embeds(sentence, attention_mask=attention_mask)
        
        # 使用 rand_init_hidden 初始化GRU的隐藏状态
        hidden = self.rand_init_hidden(batch_size)  # 这里直接初始化为张量
        if embeds[0].is_cuda:
            hidden = hidden.cuda()  # 如果使用GPU，确保hidden也转移到GPU
        
        gru_out, hidden = self.gru(embeds[0], hidden)  # GRU的输出

        # 使用自注意力机制处理GRU的输出
        attention_out, attention_weights = self.attention(gru_out)

        attention_out = attention_out.contiguous().view(-1, self.hidden_dim * 2)
        d_gru_out = self.dropout1(attention_out)
        l_out = self.liner(d_gru_out)
        gru_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return gru_feats


    def loss(self, feats, mask, tags):
        """
        计算损失
        feats: size=(batch_size, seq_len, tag_size)
        mask: size=(batch_size, seq_len)
        tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = -self.crf(feats, tags, mask)  # 使用CRF计算损失
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value
