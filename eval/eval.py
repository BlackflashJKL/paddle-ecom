import argparse
import json


def parse_arguments(parser):
    parser.add_argument('--gold_file', type=str,
                        default="../data/ECOB-ZH/test.ann.json")
    parser.add_argument('--pred_file', type=str,
                        default='../result/chinese_result/pair_classification.pred.json')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def evaluate(pred_file, gold_file):
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_opinions = json.load(f)
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_opinions = json.load(f)

    # Compute Task1_F
    pred_only_opinions = []
    for pred_opinion in pred_opinions:
        opinion = pred_opinion.copy()
        # opinion.pop('argument')
        pred_only_opinions.append(opinion)

    gold_only_opinions = []
    for gold_opinion in gold_opinions:
        opinion = gold_opinion.copy()
        # print(opinion)
        opinion.pop('argument')
        gold_only_opinions.append(opinion)

    # 任务1的F1分数
    correct_num = 0
    for gold_only_opinion in gold_only_opinions:
        if gold_only_opinion in pred_only_opinions:
            correct_num += 1
    print(correct_num, len(pred_opinions))
    p = correct_num / len(pred_opinions)
    r = correct_num / len(gold_opinions)
    f = 2 * p * r / (p + r)
    print('Task1_p', p)
    print('Task1_r', r)
    print('Task1_F: ', f)

    # Compute Task_F
    # 任务2的F1分数
    correct_num = 0
    for gold_opinion in gold_opinions:
        if gold_opinion in pred_opinions:
            correct_num += 1
    p = correct_num / len(pred_opinions)
    r = correct_num / len(gold_opinions)
    f = 2 * p * r / (p + r)
    print('Task_F: ', f)


if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    pred_file = args.pred_file
    gold_file = args.gold_file
    evaluate(pred_file, gold_file)
