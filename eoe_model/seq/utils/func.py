import json
import random

import paddle
import numpy as np
from termcolor import colored
from paddle import optimizer

from instance import Instance
from typing import List


def read_data(file: str, number: int = 5) -> List[Instance]:
    print("Reading " + file + " file.")
    insts = []

    # Read document.
    with open(file + '.doc.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)

    # Read annotation.
    try:
        with open(file + '.ann.json', 'r', encoding='utf-8') as f:
            opinions = json.load(f)
    except FileNotFoundError:
        opinions = []
        print(colored('[There is no ' + file + '.ann.json.]', 'red'))

    for doc in docs:
        event = doc['Descriptor']['text']
        event_id = doc['Descriptor']['event_id']
        doc_id = doc['Doc']['doc_id']
        title = doc['Doc']['title']
        contents = doc['Doc']['content']
        sents = [content['sent_text'] for content in contents]  # 句子列表
        labels = ['O'] * len(sents)
        targets = ['O'] * len(sents)
        doc_opinions = [opinion for opinion in opinions if int(
            opinion['doc_id']) == int(doc_id)]
        for opinion in doc_opinions:
            for sent_idx in range(opinion['start_sent_idx'], opinion['end_sent_idx'] + 1):
                labels[sent_idx] = 'I'
                targets[sent_idx] = opinion['argument']
            labels[opinion['start_sent_idx']] = 'B'
        inst = Instance(doc_id, sents, event, event_id, title, labels, targets)
        insts.append(inst)  # inst是以文章为单位的

    if number > 0:
        insts = insts[:number]

    print("Number of documents: {}".format(len(insts)))
    return insts


def log_sum_exp_paddle(vec: paddle.Tensor) -> paddle.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores = paddle.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.reshape([vec.shape[0], 1, vec.shape[2]]).expand(
        [vec.shape[0], vec.shape[1], vec.shape[2]])
    return maxScores + paddle.log(paddle.sum(paddle.exp(vec - maxScoresExpanded), 1))


def batching_list_instances(batch_size, insts: List[Instance], shffule=True):
    """
    List of instances -> List of batches
    """
    if shffule:
        insts.sort(key=lambda x: len(x.input))
    
    train_num = len(insts)
    total_batch = train_num // batch_size + \
        1 if train_num % batch_size != 0 else train_num // batch_size # 总批次数
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id *
                                batch_size:(batch_id + 1) * batch_size]
        batched_data.append(one_batch_insts)
    
    if shffule:
        random.shuffle(batched_data)
    return batched_data


def simple_batching(config, insts, tokenizer, word_pad_idx=0):
    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return
        sent_seq_len: Shape: (batch_size), the length of each paragraph in a batch.
        sent_tensor: Shape: (batch_size, max_seq_len, max_token_num)
        doc_feature: (batch_size, max_token_num)
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_data = insts
    batch_size = len(batch_data)
    word_pad_idx = word_pad_idx
    
    # doc len
    doc_sents = [inst.input for inst in insts]  # doc_num * doc_len
    events = [inst.event for inst in insts]
    max_sent_len = max([len(doc_sent) for doc_sent in doc_sents])
    # 对后面列表的每个元素都执行lambda函数中的操作，并返回一个迭代器对象
    sent_seq_len = paddle.to_tensor(
        list(map(lambda inst: len(inst.input), batch_data)))
    
    
    # sent tensor
    doc_sent_ids = []
    max_token_len = 0
    for idx, doc in enumerate(doc_sents):
        # doc_len * token_num
        sent_ids = [tokenizer(
            sent).input_ids for sent in doc] # doc_len * token_num #已解决 可以改进的地方，每个sent都和event拼接后编码了。所以改进后的模型这里的拼接是冗余的，需要删掉
        max_token_len = max(max_token_len, max(
            [len(sent) for sent in sent_ids]))
        doc_sent_ids.append(sent_ids)
    # doc features
    doc_feature_ids = [tokenizer(
            event).input_ids for event in events]
    #padding sent : (batch_size, max_seq_len, max_token_len)
    for doc_idx, doc_sent_id in enumerate(doc_sent_ids):
        for sent_idx, sent_id in enumerate(doc_sent_id):
            pad_token_num = - len(sent_id) + max_token_len
            doc_sent_ids[doc_idx][sent_idx].extend(
                [word_pad_idx]*pad_token_num)
        pad_sent_num = max_sent_len - len(doc_sent_id)
        for i in range(pad_sent_num):
            doc_sent_ids[doc_idx].append([word_pad_idx]*max_token_len) # 各个文章句子数量不同，句子也用[0,0,0..0]来填充
    #padding doc feature : (batch_size, max_token_len)
    for feature_idx, feature_id in enumerate(doc_feature_ids):
        pad_token_num = -len(feature_id) + max_token_len
        doc_feature_ids[feature_idx].extend([word_pad_idx]*pad_token_num)
        

    # label seq tensor
    label_seq_tensor = paddle.zeros(
        (batch_size, max_sent_len), dtype='int64')
    for idx in range(batch_size):
        if batch_data[idx].output_ids:
            label_seq_tensor[idx, :sent_seq_len[idx]] = paddle.to_tensor(
                batch_data[idx].output_ids)
            

    # list to tensor
    sent_tensor = paddle.to_tensor(doc_sent_ids)
    doc_feature = paddle.to_tensor(doc_feature_ids)
    label_seq_tensor = label_seq_tensor
    sent_seq_len = sent_seq_len

    return sent_seq_len, sent_tensor, doc_feature, label_seq_tensor


def lr_decay(config, optimizer: optimizer.Optimizer, epoch: int) -> optimizer.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def get_optimizer(config, model):  # 设置bertmodel学习率为backbone_lr，其他模型学习率为lr
    """
    Method to get optimizer.
    """
    params = model.parameters()

    if config.optimizer.lower() == "sgd":
        print(colored("Using SGD: lr is: {}, L2 regularization is: {}".format(
            config.learning_rate, config.l2), 'yellow'))
        return optimizer.SGD(learning_rate=config.backbone_lr, parameters=params)
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optimizer.Adam(learning_rate=config.backbone_lr, parameters=params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def use_ibo(label):
    if label.startswith('E'):
        label = 'I'
    elif label.startswith('S'):
        label = 'B'
    return label


def write_results(filename: str, insts):
    """
    Save results.
    Each json instance is an opinion.
    """
    opinions = []
    for inst in insts:
        event_id = inst.event_id
        doc_id = inst.doc_id
        labels = inst.prediction # 保存预测结果

        start_sent_idx = -1
        end_sent_idx = -1
        for sent_idx, label in enumerate(labels):
            if label == 'E':
                label = 'I'
            if label == 'S':
                label = 'B'
            if label == 'B':
                if start_sent_idx != -1:
                    opinion = {'event_id': event_id, 'doc_id': doc_id, 'start_sent_idx': start_sent_idx,
                               'end_sent_idx': end_sent_idx}
                    opinions.append(opinion)
                start_sent_idx = sent_idx
                end_sent_idx = sent_idx
            elif label == 'I':
                end_sent_idx = sent_idx
            elif label == 'O':
                if start_sent_idx != -1 and start_sent_idx <= end_sent_idx:
                    opinion = {'event_id': event_id, 'doc_id': doc_id, 'start_sent_idx': start_sent_idx,
                               'end_sent_idx': end_sent_idx}
                    opinions.append(opinion)
                start_sent_idx = -1
                end_sent_idx = -1
        if start_sent_idx != -1 and start_sent_idx <= end_sent_idx:
            opinion = {'event_id': event_id, 'doc_id': doc_id, 'start_sent_idx': start_sent_idx,
                       'end_sent_idx': end_sent_idx}
            opinions.append(opinion)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(opinions, ensure_ascii=False))

def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out