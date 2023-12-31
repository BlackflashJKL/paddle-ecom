import argparse
import random
import time
import os
import pickle
import tarfile
import paddle
import numpy as np
import logging
from typing import List
from termcolor import colored
from tqdm import tqdm
from paddlenlp.transformers import BertTokenizer
from utils import Instance
from utils import lr_decay, batching_list_instances, simple_batching, get_optimizer, set_seed, write_results, evaluate_batch_insts, read_data
from config import Config
from model import NNCRF

import faulthandler

faulthandler.enable()

def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--device', type=str, default="gpu:0",
                        choices=['cpu', 'gpu:0', 'gpu:1', 'gpu:2', 'gpu:3', 'gpu:4', 'gpu:5', 'gpu:6', 'gpu:7'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--data_dir', type=str, default="../../data/")
    parser.add_argument('--result_dir', type=str, default="../../result/chinese_result/")
    parser.add_argument('--train_file', type=str, default="train")
    parser.add_argument('--dev_file', type=str, default="dev")
    parser.add_argument('--test_file', type=str, default="test")
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--bert', type=str, default='bert-base-chinese')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1, help="default batch size is 1")
    parser.add_argument('--num_epochs', type=int, default=50, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_no_incre', type=int, default=10,
                        help="early stop when there is n epoch not increasing on dev")

    # model hyperparameter
    parser.add_argument('--model_dir', type=str, default="../../model_files/")
    parser.add_argument('--model_folder', type=str, default="seq", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=0, choices=[0, 1], help="use character-level lstm, 0 or 1")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, train_insts=None, dev_insts=None, test_insts=None):
    # Init model using config.
    model = NNCRF(config) # 自定义的NNCRF
    tokenizer = BertTokenizer.from_pretrained(config.bert, do_lower_case=True) # 调用huggingface的tokenizer

    # Count the number of parameters.
    num_param = 0
    for idx in list(model.parameters()):
        try:
            num_param += idx.shape[0] * idx.shape[1]
        except IndexError:
            num_param += idx.shape[0]
    print(num_param)

    # Get optimizer.
    optimizer = get_optimizer(config, model) # 根据给出的参数获取响应的优化器

    # Get instances.
    train_num = len(train_insts)
    print("number of instances: %d" % train_num)
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)

    batched_data = batching_list_instances(config.batch_size, train_insts) # 将insts（每个inst对应一篇文章）分批
    dev_batches = batching_list_instances(config.batch_size, dev_insts, shffule=False)
    test_batches = batching_list_instances(config.batch_size, test_insts, shffule=False)

    best_dev = [-1, 0, -1]
    best_test = [-1, 0, -1]

    # Path to save op_extract_model.
    model_folder = config.model_folder

    model_dir = config.model_dir + model_folder
    model_path = model_dir + f"/lstm_crf.m"
    config_path = model_dir + f"/config.conf"
    res_path = config.result_dir + f"{model_folder}.pred.json"

    # If model exists, evaluate and save results.
    if os.path.exists(model_dir):
        print(f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
              f"to avoid override.")
        load_layer_state_dict = paddle.load(model_path) # 读取模型参数
        model.set_state_dict(load_layer_state_dict) # 加载模型参数
        model.eval()
        evaluate_model(config, model, dev_batches, "dev", dev_insts, tokenizer)
        evaluate_model(config, model, test_batches, "test", test_insts, tokenizer)
        write_results(res_path, test_insts)
        return

    # If model not exists.
    # Create new dirs.
    print("[Info] The model will be saved to: %s.tar.gz" % model_folder)
    os.makedirs(model_dir, exist_ok=True)  # create model files. not raise error if exist.

    # Train model.
    no_incre_dev = 0
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        optimizer.clear_grad()
        if config.optimizer.lower() == "sgd": # 因为选择adam，这句暂时没用
            optimizer = lr_decay(config, optimizer, i)
        logging.info("length of batch_data is: %s" % len(batched_data))
        for index in tqdm(np.random.permutation(len(batched_data))):
            processed_batched_data = simple_batching(config, batched_data[index], tokenizer)
            model.train()
            loss = model(*processed_batched_data)
            epoch_loss += loss.item()
            logging.info("Loss is %s"%loss.item())
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts, tokenizer)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts, tokenizer)
        if dev_metrics[2] > best_dev[0]:
            print("saving the best model...")
            no_incre_dev = 0
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
            paddle.save(model.state_dict(), model_path)
            # Save the corresponding config as well.
            f = open(config_path, 'wb')
            pickle.dump(config, f)
            f.close()
            write_results(res_path, test_insts)
        else:
            no_incre_dev += 1

        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev" % no_incre_dev)
            break

    # Save best model.
    print("Archiving the best Model...")
    with tarfile.open(model_dir + f"/{model_folder}.tar.gz", "w:gz") as tar:
        tar.add(model_dir, arcname=os.path.basename(model_folder))
    print("Finished archiving the op_extract_model")

    # Print best result while training.
    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))

    # Begin test.
    print("Final testing.")
    load_layer_state_dict = paddle.load(model_path) # 读取模型参数
    model.set_state_dict(load_layer_state_dict) # 加载模型参数
    model.eval()
    evaluate_model(config, model, test_batches, "test", test_insts, tokenizer)
    write_results(res_path, test_insts)


def evaluate_model(config: Config, model: NNCRF, batch_insts_ids, name: str, insts: List[Instance], tokenizer):
    print(colored('[Begin evaluating ' + name + '.]', 'green'))
    metrics, metrics_e2e = np.asarray([0, 0, 0], dtype='int'), np.zeros((1, 3), dtype='int')
    batch_idx = 0
    batch_size = config.batch_size

    # Calculate metrics by batch.
    with paddle.no_grad():
        for batch in tqdm(batch_insts_ids):
            # get instances list.
            one_batch_insts = insts[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            # get ner result.
            processed_batched_data = simple_batching(config, batch, tokenizer)
            batch_max_scores, batch_max_ids = model.decode(processed_batched_data)

            # evaluate ner result.
            # get the num of correctly predicted arguments, predicted arguments and gold arguments.
            metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, processed_batched_data[-1],
                                            processed_batched_data[0], config.idx2labels)

            batch_idx += 1
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    print("Opinion Extraction: [%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
    print("Correctly predicted opinion: %d, Total predicted opinion: %d, Total golden opinion: % d" % (p, total_predict, total_entity))
    return [precision, recall, fscore]


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    
    logging.basicConfig(filename='scripts/log',filemode='w',level=logging.INFO)

    # update
    opt.train_file = opt.data_dir + opt.train_file
    opt.dev_file = opt.data_dir + opt.dev_file
    opt.test_file = opt.data_dir + opt.test_file
    conf = Config(opt) # 将参数、idx转换表等都保存到config类中

    set_seed(conf.seed)
    
    paddle.set_device(conf.device) # 设置全局的设备

    # Read train/test/dev data into instance.
    devs = read_data(conf.dev_file, conf.dev_num)
    tests = read_data(conf.test_file, conf.test_num)
    trains = read_data(conf.train_file, conf.train_num)

    # Data Preprocess.
    # Relabel IBO labels to IOBES labels.
    '''
    IOBES: An alternative to the IOB scheme is IOBES, 
    which increases the amount of information related to the boundaries of named entities. 
    In addition to tagging words at the beginning (B), inside (I), end (E), and outside (O) of a named entity. 
    It also labels single-token entities with the tag S.
    '''
    trains = conf.use_iobes(trains)
    devs = conf.use_iobes(devs)
    tests = conf.use_iobes(tests)
    
    conf.build_label_idx(trains + devs + tests)
    conf.build_word_idx(trains, devs, tests)

    conf.map_insts_ids(trains) # 将句子、词、句子标签全部转化为id表示
    conf.map_insts_ids(devs)
    conf.map_insts_ids(tests)

    print("num chars: " + str(conf.num_char))
    print("num words: " + str(len(conf.word2idx)))

    # Train model.
    train_model(conf, conf.num_epochs, trains, devs, tests)


if __name__ == "__main__":
    """
    command:
    python eoe_model/seq/main.py \
       --lr 5e-4 \
       --backbone_lr 1e-6 \
       --batch_size 1 \
       --bert bert-base-chinese \
       --num_epochs 10 \
       --data_dir data/ECOB-ZH/ \
       --model_dir model_files/chinese_model/ \
       --result_dir result/chinese_result/
    """
    main()
