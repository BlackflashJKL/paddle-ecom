import paddle.nn as nn
import paddle
from paddlenlp.transformers import BertModel


class BertAttentionEncoder(nn.Layer):

    def __init__(self, config):
        super(BertAttentionEncoder, self).__init__()
        # parameters
        self.num_layers = 1
        self.num_heads = config.num_heads
        self.label_size = config.label_size
        self.input_size = config.embedding_dim

        self.device = config.device

        self.label2idx = config.label2idx
        self.labels = config.idx2labels

        final_hidden_dim = config.hidden_dim

        # 创建参数,并指定参数的学习率
        '''
        weight_ih_attr (ParamAttr,可选) - weight_ih 的参数。默认为 None。
        weight_hh_attr (ParamAttr,可选) - weight_hh 的参数。默认为 None。
        bias_ih_attr (ParamAttr,可选) - bias_ih 的参数。默认为 None。
        bias_hh_attr (ParamAttr,可选) - bias_hh 的参数。默认为 None。
        '''
        self.weight_ih_attr = paddle.ParamAttr(
            learning_rate=config.lr_proportion, trainable=True)
        self.weight_hh_attr = paddle.ParamAttr(
            learning_rate=config.lr_proportion, trainable=True)
        self.bias_ih_attr = paddle.ParamAttr(
            learning_rate=config.lr_proportion, trainable=True)
        self.bias_hh_attr = paddle.ParamAttr(
            learning_rate=config.lr_proportion, trainable=True)
        self.weight_attr = paddle.ParamAttr(
            name="weight", learning_rate=config.lr_proportion, trainable=True)
        self.bias_attr = paddle.ParamAttr(
            name="bias", learning_rate=config.lr_proportion, trainable=True)

        """
        weight_attr (ParamAttr,可选) - 指定权重参数属性的对象。默认值:None,表示使用默认的权重参数属性。具体用法请参见 ParamAttr。
        bias_attr (ParamAttr,可选）- 指定偏置参数属性的对象。默认值:None,表示使用默认的偏置参数属性。具体用法请参见 ParamAttr.
        """
        self.attention_weight_attr = paddle.ParamAttr(
            learning_rate=config.lr_proportion, trainable=True)
        self.attention_bias_attr = paddle.ParamAttr(
            learning_rate=config.lr_proportion, trainable=True)

        # model
        self.bert = BertModel.from_pretrained(config.bert)

        self.attention = nn.MultiHeadAttention(
            embed_dim=self.input_size, num_heads=self.num_heads, weight_attr=self.attention_weight_attr, bias_attr=self.attention_bias_attr, dropout=config.dropout)  # cross_attention和lstm使用同样的dropout

        self.lstm = nn.LSTM(self.input_size,
                            config.hidden_dim // 2,
                            num_layers=2,
                            direction='bidirectional',
                            weight_ih_attr=self.weight_ih_attr,
                            weight_hh_attr=self.weight_hh_attr,
                            bias_ih_attr=self.bias_ih_attr,
                            bias_hh_attr=self.bias_hh_attr
                            )

        self.word_drop = nn.Dropout(config.dropout)

        self.drop_lstm = nn.Dropout(config.dropout)

        self.feature2tag = nn.Linear(
            final_hidden_dim+self.input_size, self.label_size, weight_attr=self.weight_attr, bias_attr=self.bias_attr)

    def forward(self, sent_seq_lens, sent_tensor, doc_feature):  # feature(batch_size, max_num_len)

        batch_sent = sent_tensor
        batch_desc = doc_feature
        # print("[INFO] batch_sent shape is: ",batch_sent.shape)
        # print("[INFO] batch_desc shape is: ",batch_desc.shape)

        # sentence embedding
        # 截断为 (batch_size, max_sent_len, max_num_len) max_sent_len是句子的最大数量
        batch_sent = batch_sent[:, :, :420]
        # 截断为 (batch_size, max_num_len)
        batch_desc = batch_desc[:, :420]

        """文章所有句子编码, 文章描述编码"""
        # (batch_size*max_sent_len, max_num_len) 句子总数,句子中词数
        # batch_sent_flatten = batch_sent.reshape([-1, batch_sent.shape[2]])
        
        # 将句子和描述拼接
        batch_desc=paddle.reshape(x=batch_desc,shape=[batch_desc.shape[0],1,batch_desc.shape[1]])
        batch_all=paddle.concat([batch_desc,batch_sent],axis=1)
        batch_all_flatten=batch_all.reshape([-1, batch_sent.shape[2]])
        
        try: # 尝试改进为句子和事件放在一起编码表示
            zero_tensor = paddle.zeros(
                batch_all_flatten.shape, dtype='int64')
            # 句子+描述编码
            batch_all_output = self.bert(batch_all_flatten, attention_mask=batch_all_flatten.greater_than(
                zero_tensor))[0]  # (batch_size*max_seq_len, max_num_len, hidden_size)
            
        except RuntimeError:
            print(batch_sent.shape)
            print(batch_desc.shape)

        # (batch_size, max_seq_len, hidden_size) 只保留句子中第一个词的隐含层输出,所以[:,0,:]
        batch_all_output = batch_all_output[:, 0, :].reshape([
            batch_all.shape[0], batch_all.shape[1], -1])
        
        # 再从输出中拆分出sent和desc的部分
        batch_sent_output=batch_all_output[:,1:,:]
        batch_desc_output=batch_all_output[:,0:1,:]

        """Sentence LSTM"""
        sent_rep = self.word_drop(batch_sent_output)

        lstm_out, _ = self.lstm(inputs=sent_rep, sequence_length=sent_seq_lens)

        lstm_feature = self.drop_lstm(lstm_out)

        """Cross-attention"""
        # TODO 这里需要用attn_mask吗，这里后面不知道为什么要加[0]
        attention_feature = self.attention(
            batch_sent_output, batch_desc_output, batch_desc_output)

        # 将LSTM和attention部分的输出拼接再通过线性层映射，得到条件随机场的输入
        """
        [INFO] lstm_feature shape is:  [1, 5, 200]
        [INFO] attention_feature shape is:  [5, 5, 768]
        """
        feature_out = paddle.concat(
            x=[lstm_feature, attention_feature], axis=2)  # 从最低维度进行拼接

        outputs = self.feature2tag(feature_out)

        return feature_out, outputs
