'''
eoe任务 seq.pred.json格式 篇章级别
{
    "event_id":668,
    "doc_id":2401,
    "start_sent_idx":10,
    "end_sent_idx":12
}

ote任务 mrc.ann.json格式 
{
    "event_id":591,
    "doc_id":2104,
    "start_sent_idx":9,
    "end_sent_idx":11,
    "argument":"特朗普将签法案限制中企"
}

test.ann.json格式
{
    "event_id": 668,
    "doc_id": 2401,
    "start_sent_idx": 10,
    "end_sent_idx": 13,
    "argument": "新疆"
}

分为句子级和篇章级两种比较方式
'''
import argparse
import json
from overlapf1 import calculate_overlap_f1

def parse_arguments(parser):
    parser.add_argument('--gold_file', type=str,
                        default="data/ECOB-ZH/test.ann.json")
    parser.add_argument('--pred_file', type=str,
                        default='result/chinese_result/pair_classification.pred.json')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def compare(pred_opinions, gold_opinions):
    # Compute Task_F
    # 任务2的F1分数
    correct_num = 0
    all_nume=0
    all_deno=0
    for i,gold_opinion in enumerate(gold_opinions):
        pred_opinion=pred_opinions[i]
        nume,deno=calculate_overlap_f1(pred_opinion["argument"],gold_opinion["argument"])
        all_nume+=nume
        all_deno+=deno
        if pred_opinion==gold_opinion: 
            correct_num+=1
    print(len(pred_opinions),len(gold_opinions))
    acc=correct_num/len(pred_opinions)
    f1=all_nume/all_deno
    
    print("accuracy {}, overlap-f1 {}".format(acc, f1))


def evaluate(pred_file, gold_file):
    with open(pred_file, 'r', encoding='utf-8') as f:
        raw_pred_opinions = json.load(f)
    with open(gold_file, 'r', encoding='utf-8') as f:
        raw_gold_opinions = json.load(f)

    """篇章级别"""
    print("********* Segment Level **************")

    pred_opinions = raw_pred_opinions
    gold_opinions = raw_gold_opinions

    compare(pred_opinions, gold_opinions)


if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    pred_file = args.pred_file
    gold_file = args.gold_file
    evaluate(pred_file, gold_file)
