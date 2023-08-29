from x2paddle import torch2paddle
import argparse
import os
import numpy as np
import random
import paddle
import paddlenlp.transformers
from paddlenlp.transformers import BertForSequenceClassification
from paddle.optimizer import AdamW
from paddlenlp.transformers.optimization import LinearDecayWithWarmup
from utils import *

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def parse_arguments(parser):
    parser.add_argument('--device', type=str, default='gpu:0', choices=[
        'gpu:0', 'cpu', 'gpu:0', 'gpu:1', 'gpu:2', 'gpu:3', 'gpu:4',
        'gpu:5', 'gpu:6', 'gpu:7'], help='GPU/CPU devices')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--result_dir', type=str, default=\
        '../../result/chinese_result/')
    parser.add_argument('--result_file', type=str, default='spanr.ann.json')
    parser.add_argument('--train_file', type=str, default='train')
    parser.add_argument('--dev_file', type=str, default='dev')
    parser.add_argument('--test_file', type=str, default='test')
    parser.add_argument('--bert', type=str, default='bert-base-cased')
    parser.add_argument('--opinion_level', type=str, default='segment',
        choices=['segment', 'sent'])
    parser.add_argument('--retrain', type=int, default=1, choices=[0, 1])
    parser.add_argument('--ratio', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=8, help=\
        'default batch size is 10 (works well)')
    parser.add_argument('--num_epochs', type=int, default=10, help=\
        'Usually we set to 10.')
    parser.add_argument('--model_folder', type=str, default=\
        '../../model_files/', help='The name to save the model files')
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ': ' + str(args.__dict__[k]))
    return args


def train_model(retrain=True, opinion_level='segment', ratio=3):
    model = BertForSequenceClassification.from_pretrained(raw_model,
        num_labels=2)
    # model
    if retrain:
        train_df = data_preprocess(data_dir + train_file, ratio=ratio,
            opinion_level=opinion_level, language=language)
        train_batches = data_batch(train_df, batch_size)
    if os.path.exists(best_model_dir) and not retrain:
        print(f'The folder ' + best_model_dir +
            " exists. We'll use it straightly.")
        model.load_state_dict(paddle.load(best_model_dir))
        return model
    elif os.path.exists(best_model_dir) and retrain:
        print(f'The folder ' + best_model_dir +
            " exists. We'll train it and use the best.")
        model.load_state_dict(paddle.load(best_model_dir))
    else:
        print(f'Begin training.')
    best_acc = 0
    clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    total_steps = train_epoches * len(train_batches)
    scheduler = LinearDecayWithWarmup(learning_rate, total_steps=\
        total_steps, warmup=0)
    optimizer = AdamW(parameters=model.parameters(), learning_rate=\
        scheduler, grad_clip=clip)
    for epoch_i in tqdm(range(1, train_epoches + 1)):
        total_train_loss = 0
        model.train()
        # optimizer = lr_decay(learning_rate, lr_decay_metric, optimizer, epoch_i
        #     )
        for step, batch in enumerate(tqdm(train_batches)):
            (input_ids, token_type_ids, attention_mask, labels, _, _, _, _,
                _, _, _) = batch
            input_ids = input_ids
            token_type_ids = token_type_ids
            attention_mask = attention_mask
            labels = torch2paddle.create_tensor(labels)
            optimizer.clear_grad()
            outputs = model(input_ids, token_type_ids=token_type_ids,
                attention_mask=attention_mask, labels=labels, return_dict=True)
            loss, logits = outputs.loss, outputs.logits
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        acc, _, _ = get_match_result(dev_file, model, opinion_level=\
            opinion_level, output=False)
        if acc > best_acc:
            best_acc = acc
            print('We update best model in epoch ' + str(epoch_i))
            print('Acc in Dev dataset now is', acc, sep=' ')
            paddle.save(model.state_dict(), best_model_dir)
        else:
            print('Acc does not improve.')
            print('Acc in Dev dataset now is', acc, sep=' ')
    return model


def get_match_result(file_name, model, opinion_level='segment', output=True):
    eval_df = data_preprocess(data_dir + file_name, ratio=999999,
        opinion_level=opinion_level, language=language)
    eval_batches = data_batch(eval_df, batch_size, shuffle=False)
    model.eval()
    opinions = []
    spans = []
    probs = []
    opinion_base_infos = set()
    
    with paddle.no_grad():
        for step, batch in enumerate(tqdm(eval_batches)):
            (input_ids, token_type_ids, attention_mask, labels, opinion,
                span, gold_aspect, event_id, doc_id, start_sent_id, end_sent_id
                ) = batch
            input_ids = input_ids
            token_type_ids = token_type_ids
            attention_mask = attention_mask
            labels = torch2paddle.create_tensor(labels)
            outputs = model(input_ids, token_type_ids=token_type_ids,
                attention_mask=attention_mask, labels=labels, return_dict=True)
            logits = outputs.logits.detach().cpu().numpy()
            probs.extend(logits[:, 1].tolist())
            spans.extend(span)
            opinions.extend(opinion)
            opinion_base_infos.update(list(zip(opinion, event_id, doc_id,
                start_sent_id, end_sent_id, gold_aspect)))
    
    pred_opinions = []
    correct_pred_num = 0
    for opinion in opinion_base_infos:
        o_probs = probs.copy()
        for idx, prob in enumerate(probs):
            if opinions[idx] != opinion[0]:
                o_probs[idx] = -9999999
        pred_aspect = spans[o_probs.index(max(o_probs))]
        gold_aspect = opinion[5]
        pred_opinion = {'event_id': opinion[1], 'doc_id': opinion[2],
            'start_sent_idx': opinion[3], 'end_sent_idx': opinion[4],
            'argument': pred_aspect}
        pred_opinions.append(pred_opinion)
        if ''.join(pred_aspect.strip().split()) == ''.join(gold_aspect.
            strip().split()):
            correct_pred_num += 1
    total = len(pred_opinions)
    
    if output:
        print('Accuracy: ', correct_pred_num / total, correct_pred_num, total)
    if output:
        with open(result_dir + result_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(pred_opinions, ensure_ascii=False)) # 这里改为输出pred_opinions，但是数据错位
            print('Writing result to: ' + result_dir + result_file)
    return correct_pred_num / total, correct_pred_num, total


if __name__ == '__main__':
    """
    python main.py --device gpu:3 --lr 1e-5 --batch_size 24 --retrain 1 --bert bert-base-chinese --opinion_level segment --ratio 3
    """
    parser = argparse.ArgumentParser(description='Pair Classification')
    args = parse_arguments(parser)
    batch_size = args.batch_size
    train_epoches = args.num_epochs
    learning_rate = args.lr
    lr_decay_metric = args.lr_decay
    retrain = args.retrain
    opinion_level = args.opinion_level
    ratio = args.ratio
    seed = args.seed
    raw_model = args.bert
    data_dir = args.data_dir
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file
    result_dir = args.result_dir
    result_file = args.result_file
    best_model_dir = args.model_folder + 'best_enum_model_' + str(batch_size
        ) + '_' + str(learning_rate) + '_' + str(ratio)
    model_dir = args.model_folder + 'enum_model_' + str(batch_size
        ) + '_' + str(learning_rate) + '_' + str(ratio)
    if 'chinese' in raw_model:
        language = 'chinese'
    else:
        language = 'english'
    set_seed(seed)
    device = args.device
    paddle.set_device(device)
    model = train_model(retrain, ratio=ratio, opinion_level=opinion_level)
    get_match_result(test_file, model, opinion_level)
