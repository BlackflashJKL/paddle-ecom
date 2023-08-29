import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel

from crf import LinearCRF
from bertattention_encoder import BertAttentionEncoder


class NNCRF(nn.Layer):

    def __init__(self, config):
        super(NNCRF, self).__init__()
        self.device = config.device
        # 经过bert对句子编码，再经过bilstm、cross-attention进一步提取特征
        self.encoder = BertAttentionEncoder(config)
        self.inferencer = LinearCRF(config)  # 最后再通过条件随机场

    def forward(self,
                sent_seq_lens: paddle.Tensor,
                sent_tensor: paddle.Tensor,
                doc_feature: paddle.Tensor,
                tags: paddle.Tensor) -> paddle.Tensor:
        """
        Calculate the negative loglikelihood.
        :return: the total negative log-likelihood loss
        """
        # Encode.
        _, lstm_scores = self.encoder(sent_seq_lens, sent_tensor, doc_feature)
        batch_size = sent_tensor.shape[0]
        sent_len = sent_tensor.shape[1]
        maskTemp = paddle.arange(
            1, sent_len + 1, dtype='int64').reshape([1, sent_len]).expand([batch_size, sent_len])
        mask = paddle.less_equal(maskTemp, sent_seq_lens.reshape(
            [batch_size, 1]).expand([batch_size, sent_len]))
        # Inference.
        unlabed_score, labeled_score = self.inferencer(
            lstm_scores, sent_seq_lens, tags, mask)
        return unlabed_score - labeled_score

    def decode(self, batchInput):
        """
        Decode the batch input
        """
        wordSeqLengths, initial_wordSeqTensor, featureSeqTensor, tagSeqTensor = batchInput
        # Encode.
        feature_out, features = self.encoder(
            wordSeqLengths, initial_wordSeqTensor, featureSeqTensor)
        # Decode.
        bestScores, decodeIdx = self.inferencer.decode(
            features, wordSeqLengths)
        return bestScores, decodeIdx
