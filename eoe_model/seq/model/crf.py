import paddle.nn as nn
import paddle

from utils import log_sum_exp_paddle, START, STOP, PAD
from utils.func import paddle_gather
from typing import Tuple


class LinearCRF(nn.Layer):

    def __init__(self, config):
        super(LinearCRF, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn

        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        self.start_idx = self.label2idx[START]
        self.end_idx = self.label2idx[STOP]
        self.pad_idx = self.label2idx[PAD]

        # initialize the following transition
        # anything never -> start. end never -> anything. Same thing for the padding label.
        init_transition = paddle.randn([self.label_size, self.label_size])
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0

        self.transition = paddle.create_parameter(attr=paddle.ParamAttr(
            name="transition", learning_rate=config.lr_proportion), shape=init_transition.shape, dtype=str(init_transition.numpy().dtype))

    def forward(self, lstm_scores, word_seq_lens, tags, mask):
        """
        Calculate the negative log-likelihood
        :param lstm_scores:
        :param word_seq_lens:
        :param tags:
        :param mask:
        :return:
        """
        all_scores = self.calculate_all_scores(lstm_scores=lstm_scores)
        unlabed_score = self.forward_unlabeled(
            all_scores, word_seq_lens)  # calculate scores of prediction
        labeled_score = self.forward_labeled(
            all_scores, word_seq_lens, tags, mask)  # calculate scores of gold labels

        return unlabed_score, labeled_score

    def forward_unlabeled(self, all_scores: paddle.Tensor, word_seq_lens: paddle.Tensor) -> paddle.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: The score for all the possible structures.
        """
        batch_size = all_scores.shape[0]
        seq_len = all_scores.shape[1]
        alpha = paddle.zeros([batch_size, seq_len, self.label_size])

        # the first position of all labels = (the transition from start - > all labels) + current emission.
        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :]

        for word_idx in range(1, seq_len):
            # batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].\
                reshape([batch_size, self.label_size, 1]).\
                expand([batch_size, self.label_size, self.label_size]) + \
                all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_paddle(before_log_sum_exp)

        # batch_size x label_size
        last_alpha = paddle_gather(alpha, 1, word_seq_lens.reshape([batch_size, 1, 1]).expand([
            batch_size, 1, self.label_size])-1).reshape([batch_size, self.label_size])
        last_alpha += self.transition[:, self.end_idx].reshape([
            1, self.label_size]).expand([batch_size, self.label_size])
        last_alpha = log_sum_exp_paddle(last_alpha.reshape([
            batch_size, self.label_size, 1])).reshape([batch_size])

        # final score for the unlabeled network in this batch, with size: 1
        return paddle.sum(last_alpha)

    def forward_labeled(self, all_scores: paddle.Tensor, word_seq_lens: paddle.Tensor, tags: paddle.Tensor, masks: paddle.Tensor) -> paddle.Tensor:
        '''
        Calculate the scores for the gold instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences Shape: (batch_size)
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        # all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = paddle_gather(all_scores, 3, tags.reshape([batchSize, sentLength, 1, 1]).expand(
            [batchSize, sentLength, self.label_size, 1])).reshape([batchSize, -1, self.label_size])
        if sentLength != 1:
            tagTransScoresMiddle = paddle_gather(
                currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].reshape([batchSize, sentLength - 1, 1])).reshape([batchSize, -1])
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = paddle_gather(
            tags, 1, word_seq_lens.reshape([batchSize, 1]) - 1)
        tagTransScoresEnd = paddle_gather(self.transition[:, self.end_idx].reshape([
            1, self.label_size]).expand([batchSize, self.label_size]), 1,  endTagIds).reshape([batchSize])
        score = paddle.sum(tagTransScoresBegin) + paddle.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += paddle.sum(
                tagTransScoresMiddle.masked_select(masks[:, 1:]))
        return score

    def backward(self, lstm_scores: paddle.Tensor, word_seq_lens: paddle.Tensor) -> paddle.Tensor:
        """
        Backward algorithm. A benchmark implementation which is ready to use.
        :param lstm_scores: shape: (batch_size, sent_len, label_size) NOTE: the score from LSTMs, not `all_scores` (which add up the transtiion)
        :param word_seq_lens: shape: (batch_size,)
        :return: Backward variable
        """
        batch_size = lstm_scores.shape[0]
        seq_len = lstm_scores.shape[1]
        beta = paddle.zeros(batch_size, seq_len, self.label_size)

        # reverse the reshape of computing the score. we look from behind
        rev_score = self.transition.transpose(0, 1).reshape([1, 1, self.label_size, self.label_size]).expand([batch_size, seq_len, self.label_size, self.label_size]) + \
            lstm_scores.reshape([batch_size, seq_len, 1, self.label_size]).expand([
                batch_size, seq_len, self.label_size, self.label_size])

        # The code below, reverse the score from [0 -> length]  to [length -> 0].
        # (NOTE: we need to avoid reversing the padding)
        perm_idx = paddle.zeros(batch_size, seq_len)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = paddle.range(
                word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]

        # backward operation
        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = beta[:, word_idx - 1, :].reshape([batch_size, self.label_size, 1]).expand([
                batch_size, self.label_size, self.label_size]) + rev_score[:, word_idx, :, :]
            beta[:, word_idx, :] = log_sum_exp_paddle(before_log_sum_exp)

        # Following code is used to check the backward beta implementation
        last_beta = paddle_gather(beta, 1, word_seq_lens.reshape([batch_size, 1, 1]).expand([
            batch_size, 1, self.label_size]) - 1).reshape([batch_size, self.label_size])
        last_beta += self.transition.transpose(0, 1)[:, self.start_idx].reshape([
            1, self.label_size]).expand([batch_size, self.label_size])
        last_beta = log_sum_exp_paddle(last_beta.reshape([
            batch_size, self.label_size, 1])).reshape([batch_size])

        # This part if optionally, if you only use `last_beta`.
        # Otherwise, you need this to reverse back if you also need to use beta
        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]

        return paddle.sum(last_beta)

    def calculate_all_scores(self, lstm_scores: paddle.Tensor) -> paddle.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param lstm_scores: emission scores.
        :return:
        """
        batch_size = lstm_scores.shape[0]
        seq_len = lstm_scores.shape[1]
        scores = self.transition.reshape([1, 1, self.label_size, self.label_size]).expand([batch_size, seq_len, self.label_size, self.label_size]) + \
            lstm_scores.reshape([batch_size, seq_len, 1, self.label_size]).expand([
                batch_size, seq_len, self.label_size, self.label_size])
        return scores

    def decode(self, features, wordSeqLengths) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbi_decode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx

    def viterbi_decode(self, all_scores: paddle.Tensor, word_seq_lens: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_seq_len x num_labels)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_seq_len)
        """
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        scoresRecord = paddle.zeros([batchSize, sentLength, self.label_size])
        idxRecord = paddle.zeros(
            [batchSize, sentLength, self.label_size], dtype='int64')
        mask = paddle.ones_like(word_seq_lens, dtype='int64')
        startIds = paddle.full((batchSize, self.label_size),
                               self.start_idx, dtype='int64')
        decodeIdx = paddle.zeros([batchSize, sentLength],dtype='int64')

        scores = all_scores
        # represent the best current score from the start, is the best
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            # scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].reshape([batchSize, self.label_size, 1]).expand([batchSize, self.label_size,
                                                                                                         self.label_size]) + scores[:, wordIdx, :, :]
            # the best previous label idx to current labels
            idxRecord[:, wordIdx, :] = paddle.argmax(scoresIdx, 1)
            scoresRecord[:, wordIdx, :] = paddle_gather(scoresIdx, 1, idxRecord[:, wordIdx, :].reshape([
                batchSize, 1, self.label_size])).reshape([batchSize, self.label_size])

        lastScores = paddle_gather(scoresRecord, 1, word_seq_lens.reshape([batchSize, 1, 1]).expand([
            batchSize, 1, self.label_size]) - 1).reshape([batchSize, self.label_size])  # select position
        lastScores += self.transition[:, self.end_idx].reshape([
            1, self.label_size]).expand([batchSize, self.label_size])
        decodeIdx[:, 0] = paddle.argmax(lastScores, 1)
        bestScores = paddle_gather(
            lastScores, 1, decodeIdx[:, 0].reshape([batchSize, 1]))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = paddle_gather(idxRecord, 1, paddle.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens -
                                           distance2Last - 1, mask).reshape([batchSize, 1, 1]).expand([batchSize, 1, self.label_size])).reshape([batchSize, self.label_size])
            decodeIdx[:, distance2Last + 1] = paddle_gather(
                lastNIdxRecord, 1, decodeIdx[:, distance2Last].reshape([batchSize, 1])).reshape([batchSize])

        return bestScores, decodeIdx
