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


def parse_arguments(parser):
    parser.add_argument('--gold_file', type=str,
                        default="data/ECOB-ZH/test.ann.json")
    parser.add_argument('--pred_file', type=str,
                        default='result/chinese_result/pair_classification.pred.json')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def toSentence(opinions):
    # 将篇章级别的json转化为句子级别
    results = []

    for opinion in opinions:
        result = opinion
        if opinion["start_sent_idx"] != opinion["end_sent_idx"]:
            for i in range(opinion["start_sent_idx"], opinion["end_sent_idx"]+1):
                result["start_sent_idx"] = i
                result["end_sent_idx"] = i
                results.append(result.copy())
        else:
            results.append(result.copy())

    return results


def toSegment(opinions):
    # 将句子级别的json转化为篇章级别
    results = []
    result = opinions[0].copy()

    for i, opinion in enumerate(opinions):
        if i != 0:
            if opinion["event_id"] == result["event_id"] and opinion["doc_id"] == result["doc_id"] and opinion["start_sent_idx"] == result["end_sent_idx"]+1 and opinion["argument"] == result["argument"]:
                result["end_sent_idx"] += opinion["end_sent_idx"]-opinion["start_sent_idx"]+1
            else:
                results.append(result.copy())
                result = opinion.copy()
                # print(result["start_sent_idx"])

    results.append(result.copy())
    return results


def compare(pred_opinions, gold_opinions):
    # Compute Task_F
    # 任务2的F1分数
    correct_num = 0
    for gold_opinion in gold_opinions:
        if gold_opinion in pred_opinions:
            correct_num += 1
    p = correct_num / len(pred_opinions)
    r = correct_num / len(gold_opinions)
    f = 2 * p * r / (p + r)
    print("precision {}, recall {}, f1 {}".format(p, r, f))


def evaluate(pred_file, gold_file):
    with open(pred_file, 'r', encoding='utf-8') as f:
        raw_pred_opinions = json.load(f)
    with open(gold_file, 'r', encoding='utf-8') as f:
        raw_gold_opinions = json.load(f)

    """篇章级别"""
    print("********* Segment Level **************")

    pred_opinions = toSegment(raw_pred_opinions)
    # pred_opinions = raw_pred_opinions
    gold_opinions = raw_gold_opinions

    # for opinion in pred_opinions:
    #     print(opinion["start_sent_idx"])
    with open("pred_seg.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_opinions, ensure_ascii=False))

    with open("gold_seg.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(gold_opinions, ensure_ascii=False, indent=2))

    compare(pred_opinions, gold_opinions)

    """句子级别"""
    print("********* Sentence Level **************")

    pred_opinions = toSentence(raw_pred_opinions)
    gold_opinions = toSentence(raw_gold_opinions)

    compare(pred_opinions, gold_opinions)


if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    pred_file = args.pred_file
    gold_file = args.gold_file
    evaluate(pred_file, gold_file)
