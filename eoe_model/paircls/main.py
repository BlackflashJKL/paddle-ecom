import argparse
import os
from pickle import FALSE
import numpy as np
import random
import paddle
import paddlenlp
from paddlenlp.transformers import BertForSequenceClassification
from paddlenlp.transformers import LinearDecayWithWarmup
from paddle.optimizer import AdamW
from utils import *


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--device', type=str, default="gpu:0",
                        help="GPU/CPU devices")
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--result_dir', type=str,
                        default='../../result/chinese_result/')
    parser.add_argument('--train_file', type=str, default="train")
    parser.add_argument('--dev_file', type=str, default="dev")
    parser.add_argument('--test_file', type=str, default="test")
    parser.add_argument('--bert', type=str, default="bert-base-cased")
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--retrain', type=int, default=1, choices=[0, 1])
    parser.add_argument('--batch_size', type=int, default=24,
                        help="default batch size is 24 (works well)")
    parser.add_argument('--num_epochs', type=int,
                        default=20, help="Usually we set to 20.")

    # model hyperparameter
    parser.add_argument('--model_folder', type=str,
                        default="../../model_files/", help="The name to save the model files")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(retrain=True):
    # Init model.
    model = BertForSequenceClassification.from_pretrained(
        raw_model, num_labels=2)  # 从huggingface.co下载名为raw_model的模型
    model.to(device)

    # Load training data.
    if retrain:
        train_df = read_data(data_dir + train_file)
        dev_df = read_data(data_dir + dev_file)
        eval_df = read_data(data_dir + test_file)

    # Get Bert input format.
    if retrain:
        train_batches = data_batch(train_df, batch_size, raw_model)
        dev_batches = data_batch(dev_df, batch_size, raw_model, shffule=False)
        eval_batches = data_batch(
            eval_df, batch_size, raw_model, shffule=False)

    # If model exists, evaluate and save results.
    if os.path.exists(best_model_dir) and not retrain:
        print(f"The folder " + best_model_dir +
              " exists. We'll use it straightly.")
        load_layer_state_dict = paddle.load(best_model_dir)  # 读取模型参数
        model.set_state_dict(load_layer_state_dict)  # 加载模型参数
        return model
    elif os.path.exists(best_model_dir) and retrain:
        print(f"The folder " + best_model_dir +
              " exists. We'll train it and use the best.")
        load_layer_state_dict = paddle.load(best_model_dir)  # 读取模型参数
        model.set_state_dict(load_layer_state_dict)  # 加载模型参数
    else:
        print(f"Begin training.")

    # Train model.
    best_acc = 0

    total_steps = train_epoches * len(train_batches)
    scheduler = LinearDecayWithWarmup(
        learning_rate, total_steps=total_steps, warmup=0)
    clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    optimizer = AdamW(parameters=model.parameters(),
                      learning_rate=scheduler, grad_clip=clip)

    for epoch_i in tqdm(range(1, train_epoches+1)):
        total_train_loss = 0
        model.train()
        # train model in train dataset.
        for step, batch in enumerate(tqdm(train_batches)):  # 这样写显示进度条
            input_ids, token_type_ids, attention_mask, labels, _, _, _, _, _ = batch
            # input_ids = input_ids.to(device)
            # token_type_ids = token_type_ids.to(device)
            # attention_mask = attention_mask.to(device)
            # labels = paddle.to_tensor(labels).to(device)
            optimizer.clear_grad()

            outputs = model(input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, labels=labels, return_dict=True)  # 自动计算了损失
            loss, logits = outputs.loss, outputs.logits
            total_train_loss += loss.item()  # 累计训练总损失
            loss.backward()  # 反向传播

            # paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # 更新参数
            scheduler.step()
        # Test model in dev dataset.
        model.eval()
        dev_gold_labels = []
        dev_pred_labels = []
        with paddle.no_grad():
            for step, batch in enumerate(tqdm(dev_batches)):
                input_ids, token_type_ids, attention_mask, labels, _, _, _, _, _ = batch
                # input_ids = input_ids.to(device)
                # token_type_ids = token_type_ids.to(device)
                # attention_mask = attention_mask.to(device)
                # labels = paddle.to_tensor(labels).to(device)

                outputs = model(input_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, labels=labels, return_dict=True)

                # logits = outputs.logits.detach().cpu().numpy()
                logits = outputs.logits.detach().numpy()
                pred_label = np.argmax(logits, axis=1).flatten()  # 获取预测的类
                dev_pred_labels.extend(pred_label)  # 全部预测结果
                # label_ids = labels.to('cpu').numpy()
                label_ids = labels.numpy()
                dev_gold_labels.extend(label_ids)  # 全部真实结果
        acc = np.sum(np.array(dev_gold_labels) == np.array(
            dev_pred_labels)) / len(dev_pred_labels)  # 计算准确率
        if acc > best_acc:
            best_acc = acc
            print('We update best model in epoch ' + str(epoch_i))
            print('Acc in Dev dataset now is', acc, sep=' ')
            paddle.save(model.state_dict(), best_model_dir)  # 保存模型参数
        else:
            print('Acc does not improve.')
            print('Acc in Dev dataset now is', acc, sep=' ')
        # paddle.save(model.state_dict(), model_dir + '_' + str(epoch_i))

    # Test model
    load_layer_state_dict = paddle.load(best_model_dir)  # 读取模型参数
    model.set_state_dict(load_layer_state_dict)  # 加载模型参数
    model.eval()
    eval_gold_labels = []
    eval_pred_labels = []
    with paddle.no_grad():
        for step, batch in tqdm(enumerate(eval_batches)):
            input_ids, token_type_ids, attention_mask, labels, _, _, _, _, _ = batch
            # input_ids = input_ids.to(device)
            # token_type_ids = token_type_ids.to(device)
            # attention_mask = attention_mask.to(device)
            # labels = paddle.to_tensor(labels).to(device)

            outputs = model(input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, labels=labels, return_dict=True)

            logits = outputs.logits.detach().numpy()
            pred_label = np.argmax(logits, axis=1).flatten()
            eval_pred_labels.extend(pred_label)
            label_ids = labels.numpy()
            eval_gold_labels.extend(label_ids)
    acc = np.sum(np.array(eval_gold_labels) == np.array(
        eval_pred_labels)) / len(eval_pred_labels)
    print('Acc in Test dataset now is', acc, sep=' ')

    return model


def get_match_result(file_name, model):
    print("begin get_match_result")
    # Init.
    eval_df = read_data(data_dir + file_name)
    eval_batches = data_batch(eval_df, batch_size, raw_model, False)

    # Predict.
    model.eval()
    eval_gold_labels = []
    eval_pred_labels = []
    sents = []
    events = []
    event_ids = []
    doc_ids = []
    sent_ids = []
    with paddle.no_grad():
        for step, batch in enumerate(tqdm(eval_batches)):
            input_ids, token_type_ids, attention_mask, labels, sent, event, event_id, doc_id, sent_id = batch
            sents.extend(sent)
            events.extend(event)
            event_ids.extend(event_id)
            doc_ids.extend(doc_id)
            sent_ids.extend(sent_id)

            # input_ids = input_ids.to(device)
            # token_type_ids = token_type_ids.to(device)
            # attention_mask = attention_mask.to(device)
            # labels = paddle.to_tensor(labels).to(device)

            outputs = model(input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, labels=labels, return_dict=True)

            logits = outputs.logits.detach().numpy()
            pred_label = np.argmax(logits, axis=1).flatten()
            eval_pred_labels.extend(pred_label)
            label_ids = labels.numpy()
            eval_gold_labels.extend(label_ids)
    acc = np.sum(np.array(eval_gold_labels) == np.array(
        eval_pred_labels)) / len(eval_pred_labels)
    print('Accuracy: ', acc)

    # Save match result
    opinions = []
    for idx, pred_label in enumerate(eval_pred_labels):
        opinion = {'event_id': event_ids[idx],
                   'doc_id': doc_ids[idx],
                   'start_sent_idx': sent_ids[idx],
                   'end_sent_idx': sent_ids[idx]}
        if pred_label:
            opinions.append(opinion)
    with open(result_dir + result_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(opinions))


if __name__ == '__main__':
    """
    python eoe_model/paircls/main.py \
       --bert bert-base-chinese \
       --lr 5e-6 \
       --batch_size 24 \
       --retrain 1 \
       --num_epochs 10 \
       --model_folder model_files/chinese_model/ \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/
    """
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Pair Classification")
    args = parse_arguments(parser)

    batch_size = args.batch_size
    train_epoches = args.num_epochs
    learning_rate = args.lr
    lr_decay_metric = args.lr_decay
    seed = args.seed

    raw_model = args.bert  # 要从huggingface.co下载的模型名称
    data_dir = args.data_dir
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file
    result_dir = args.result_dir
    result_file = 'pair_classification.pred.json'
    best_model_dir = args.model_folder + 'best_pair_cls_model_' + \
        str(batch_size) + '_' + str(learning_rate) + '_' + str(lr_decay_metric)
    model_dir = args.model_folder + 'pair_cls_model_' + \
        str(batch_size) + '_' + str(learning_rate) + '_' + str(lr_decay_metric)

    retrain = args.retrain
    # Set seed
    set_seed(seed)
    # Get model
    device = args.device
    paddle.set_device(device)
    model = train_model(retrain=retrain)
    # Get result 获取测试输出结果
    get_match_result(test_file, model)
