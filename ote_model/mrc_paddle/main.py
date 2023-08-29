import argparse
import json
import logging
import os
import random
import numpy as np
import paddle
import collections
from x2paddle.torch2paddle import DataLoader
from tqdm import tqdm
from tqdm import trange
from model import RobertaForQuestionAnswering
# from paddlenlp.transformers import RobertaForQuestionAnswering
from paddlenlp.transformers import RobertaModel, RobertaTokenizer, AutoTokenizer
from paddlenlp.transformers import XLMModel
from paddlenlp.transformers import XLMForQuestionAnsweringSimple
from paddlenlp.transformers import XLMTokenizer
from paddlenlp.transformers import XLNetModel
from paddlenlp.transformers import XLNetForQuestionAnswering
from paddlenlp.transformers import XLNetTokenizer
from paddle.optimizer import AdamW
from paddlenlp.transformers.optimization import LinearDecayWithWarmup
from utils import read_squad_examples
from utils import convert_examples_to_features
from utils import write_predictions
from utils import write_predictions_extended
from paddle.nn import ClipGradByNorm
from paddle.io import Dataset, Sampler, TensorDataset, RandomSampler, SequenceSampler
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
MODEL_CLASSES = {'bert': (RobertaModel, RobertaForQuestionAnswering,
                          AutoTokenizer), 'xlnet': (XLNetModel, XLNetForQuestionAnswering,
                                                    XLNetTokenizer), 'xlm': (XLMModel, XLMForQuestionAnsweringSimple, XLMTokenizer)}

RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits", "span_logits"])
RawResultExtended = collections.namedtuple("RawResultExtended",
                                           ["unique_id", "start_top_log_probs", "start_top_index",
                                            "end_top_log_probs", "end_top_index", "cls_logits"])


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    t_total = len(train_dataloader
                  ) // args.gradient_accumulation_steps * args.num_train_epochs

    clip = paddle.nn.ClipGradByNorm(clip_norm=args.max_grad_norm)
    scheduler = LinearDecayWithWarmup(learning_rate=args.learning_rate, warmup=args.
                                      warmup_steps, total_steps=t_total)
    optimizer = AdamW(parameters=model.parameters(), learning_rate=scheduler, epsilon=args.
                      adam_epsilon, grad_clip=clip)

    logger.info('***** Running training *****')
    logger.info('  Num examples = %d', len(train_dataset))
    logger.info('  Num Epochs = %d', args.num_train_epochs)
    logger.info('  Instantaneous batch size per GPU = %d', args.
                per_gpu_train_batch_size)
    logger.info(
        '  Total train batch size (w. parallel, distributed & accumulation) = %d', args.train_batch_size * args.gradient_accumulation_steps)
    logger.info('  Gradient Accumulation steps = %d', args.
                gradient_accumulation_steps)
    logger.info('  Total optimization steps = %d', t_total)
    global_step = 0
    tr_loss = 0.0
    best_precision = 0
    results = dict()
    results['precision'] = 0
    optimizer.clear_grad()
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch',
                            disable=False)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc='Iteration', disable=False
                              )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else
                      batch[2], 'start_positions': batch[3], 'end_positions':
                      batch[4]}
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5], 'p_mask': batch[6]})
            outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                optimizer.step()
                optimizer.clear_grad()
                global_step += 1
                if (args.logging_steps > 0 and global_step % args.
                        logging_steps == 0 and args.evaluate_during_training):
                    results = evaluate(args, model, tokenizer)
                    if results['precision'] > best_precision:
                        best_precision = results['precision']
                        output_dir = os.path.join(args.output_dir)
                        model_to_save = model.module if hasattr(model, 'module'
                                                                ) else model
                        model_to_save.save_pretrained(output_dir)
                        paddle.save(args, os.path.join(output_dir,
                                                       'training_args.bin'))
                        logger.info('Saving best model to %s', output_dir)
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=''):
    dataset, examples, features = load_and_cache_examples(args, tokenizer,
                                                          evaluate=True, output_examples=True)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequenceSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    logger.info('***** Running evaluation {} *****'.format(prefix))
    logger.info('  Num examples = %d', len(dataset))
    logger.info('  Batch size = %d', args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with paddle.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else
                      batch[2]}
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4], 'p_mask': batch[5]})
            outputs = model(**inputs) # 没提供label就不返回loss
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']: # 这个分支可以暂时忽略
                result = RawResultExtended(unique_id=unique_id,
                                           start_top_log_probs=to_list(
                                               outputs[0][i]),
                                           start_top_index=to_list(
                                               outputs[1][i]),
                                           end_top_log_probs=to_list(outputs[2][i]), end_top_index=to_list(outputs[3][i]), cls_logits=to_list(outputs[4][i]))
            else:
                # print("[INFO] outputs[0] shape is {}".format(outputs[0].shape))
                # print("[INFO] outputs[1] shape is {}".format(outputs[1].shape))
                # print("[INFO] outputs[2] shape is {}".format(outputs[2].shape))
                result = RawResult(unique_id=unique_id, start_logits=to_list(
                    outputs[0][i]), end_logits=to_list(outputs[1][i]), span_logits=to_list(outputs[2][i]))
            all_results.append(result)
    output_prediction_file = os.path.join(args.output_dir,
                                          'predictions_{}.json'.format(prefix))
    output_nbest_file = os.path.join(args.output_dir,
                                     'nbest_predictions_{}.json'.format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir,
                                                 'null_odds_{}.json'.format(prefix))
    else:
        output_null_log_odds_file = None
    if args.model_type in ['xlnet', 'xlm']:
        write_predictions_extended(examples, features, all_results, args.
                                   n_best_size, args.max_answer_length, output_prediction_file,
                                   output_nbest_file, output_null_log_odds_file, args.predict_file,
                                   model.config.start_n_top, model.config.end_n_top, args.
                                   version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        all_predictions = write_predictions(examples, features, all_results,
                                            args.n_best_size, args.max_answer_length, args.do_lower_case,
                                            output_prediction_file, output_nbest_file,
                                            output_null_log_odds_file, args.verbose_logging, args.
                                            version_2_with_negative, args.null_score_diff_threshold)

        def get_clear_text(text):
            text = text.strip().split()
            return ''.join(text)
        corr_num = 0
        for example in examples:
            idx = example.qas_id
            pred_aspect = all_predictions[idx]
            gold_aspect = example.orig_answer_text
            if get_clear_text(pred_aspect) == get_clear_text(gold_aspect):
                corr_num += 1
        result = dict()
        result['precision'] = corr_num / len(examples)
        output_file = os.path.join(args.result_dir, args.result_file.format
                                   (prefix))
        opinions = []
        for example in examples:
            if args.language == 'english':
                opinion = {'event_id': example.event_id, 'doc_id': example.
                           doc_id, 'start_sent_idx': example.start_sent_idx,
                           'end_sent_idx': example.end_sent_idx, 'argument':
                           all_predictions[example.qas_id]}
            else:
                opinion = {'event_id': example.event_id, 'doc_id': example.
                           doc_id, 'start_sent_idx': example.start_sent_idx,
                           'end_sent_idx': example.end_sent_idx, 'argument':
                           get_clear_text(all_predictions[example.qas_id])}
            opinions.append(opinion)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(opinions, ensure_ascii=False))
            logger.info('Writing result to: ' + output_file)
        
        logger.info('Eval Precision: ' + str(corr_num / len(examples)) +
                    '\t' + str(corr_num) + '\t' + str(len(examples)))
    return result


def load_and_cache_examples(args, tokenizer, evaluate=False,
                            output_examples=False):
    input_file = (args.data_dir + args.predict_file if evaluate else args.
                  data_dir + args.train_file)
    cached_features_file = os.path.join(os.path.dirname(input_file),
                                        'cached_{}_{}_{}'.format('dev' if evaluate else 'train', list(
                                            filter(None, args.model_name_or_path.split('/'))).pop(), str(args.
                                                                                                         max_seq_length)))
    if os.path.exists(cached_features_file
                      ) and not args.overwrite_cache and not output_examples:
        logger.info('Loading features from cached file %s',
                    cached_features_file)
        features = paddle.load(cached_features_file)
    else:
        logger.info('Creating features from dataset file at %s', input_file)
        examples = read_squad_examples(input_file=input_file, language=args
                                       .language)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer, max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride, max_query_length=args.
                                                max_query_length, is_training=not evaluate)
        logger.info('Saving features into cached file %s', cached_features_file
                    )
        paddle.save(features, cached_features_file)
    all_input_ids = paddle.to_tensor(
        [f.input_ids for f in features], dtype='int64')
    all_input_mask = paddle.to_tensor([f.input_mask for f in features],
                                      dtype='int64')
    all_segment_ids = paddle.to_tensor([f.segment_ids for f in features],
                                       dtype='int64')
    all_cls_index = paddle.to_tensor(
        [f.cls_index for f in features], dtype='int64')
    all_p_mask = paddle.to_tensor(
        [f.p_mask for f in features], dtype='float32')
    if evaluate:
        all_example_index = paddle.arange(all_input_ids.size(
            0), dtype=paddle.int64).requires_grad_(False)
        dataset = TensorDataset(
            [all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_cls_index, all_p_mask])
    else:
        all_start_positions = paddle.to_tensor([f.start_position for f in
                                                features], dtype='int64')
        all_end_positions = paddle.to_tensor([f.end_position for f in
                                              features], dtype='int64')
        dataset = TensorDataset([all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions, all_cls_index, all_p_mask])
    if output_examples:
        return dataset, examples, features
    return dataset


def parse_arguments(parser):
    parser.add_argument('--model_type', default='bert', type=str,
                        help='Model type selected in the list: ' + ', '.join(MODEL_CLASSES.keys()))
    parser.add_argument('--model_name_or_path', default=None, type=str,
                        required=True)
    parser.add_argument('--tokenizer_name', default='', type=str,
                        help='Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--num_train_epochs', default=3.0, type=float,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--per_gpu_train_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for training.')
    parser.add_argument('--per_gpu_eval_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for evaluation.')
    parser.add_argument('--learning_rate', default=5e-05,
                        type=float, help='The initial learning rate for Adam.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.'
                        )
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, help='Max gradient norm.')
    parser.add_argument('--warmup_steps', default=0, type=int,
                        help='Linear warmup over warmup_steps.')
    parser.add_argument('--adam_epsilon', default=1e-08,
                        type=float, help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_seq_length', default=512, type=int, help='The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.'
                        )
    parser.add_argument('--doc_stride', default=128, type=int, help='When splitting up a long document into chunks, how much stride to take between chunks.'
                        )
    parser.add_argument('--max_query_length', default=384, type=int, help='The maximum number of tokens for the question. Questions longer than this will be truncated to this length.'
                        )
    parser.add_argument('--max_answer_length', default=30, type=int, help='The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.'
                        )
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Set this flag if you are using an uncased model.')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--data_dir', default='data/ECOB-EN/', type=str)
    parser.add_argument('--train_file', default='train',
                        type=str, help='Training file.')
    parser.add_argument('--predict_file', default='dev',
                        type=str, help='Predicting file.')
    parser.add_argument('--result_dir', default='result/english_result/',
                        type=str)
    parser.add_argument('--result_file', default='mrc.ann.json', type=str)
    parser.add_argument('--output_dir', default='model_files/bert_sqad/',
                        type=str, help='The output directory where the model checkpoints and predictions will be written.'
                        )
    parser.add_argument('--cache_dir', default='', type=str,
                        help='Where do you want to store the pre-trained models downloaded from s3')
    parser.add_argument('--config_name', default='', type=str,
                        help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0, help='If null_score - best_non_null is greater than the threshold predict null.'
                        )
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--evaluate_during_training', action='store_true',
                        help='Rul evaluation during training at each logging step.')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu',
                                                                      'gpu:0', 'gpu:1', 'gpu:2', 'gpu:3', 'gpu:4', 'gpu:5',
                                                                      'gpu:6', 'gpu:7'], help='GPU/CPU devices')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--fp16', action='store_true', help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit'
                        )
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at https://nvidia.github.io/apex/amp.html"
                        )
    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.') #TODO 这个参数不知道是干什么的
    parser.add_argument('--logging_steps', type=int,
                        default=100, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every X updates steps.')
    parser.add_argument('--n_best_size', default=20, type=int, help='The total number of n-best predictions to generate in the nbest_predictions.json output file.'
                        )
    parser.add_argument('--verbose_logging', action='store_true', help='If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.'
                        )
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help='Overwrite the content of the output directory')
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ': ' + str(args.__dict__[k]))
    return args


def main():
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir
                                                      ) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            'Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.'
            .format(args.output_dir))
    set_seed(args)
    paddle.set_device(args.device)
    
    # 加载模型
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    print("****************",config_class,model_class,tokenizer_class,args.config_name if args.config_name else args.model_name_or_path)
    config = config_class.from_pretrained(args.config_name if args.
                                          config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args
                                                .tokenizer_name else args.model_name_or_path, do_lower_case=args.
                                                do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(
        '.pdiparams' in args.model_name_or_path), config=config)
    # model = model_class.from_pretrained(args.model_name_or_path)
    # config_class=RobertaConfig()
    # model = RobertaForQuestionAnswering.from_pretrained()
    """
    这里报错是因为paddlenlp版本问题,需要用旧版2.4,新版2.5用的是RobertaConfig而不是RobertaModel
    """
    # 训练
    model.to(args.device)
    logger.info('Training/evaluation parameters %s', args)
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(' global_step = %s, average loss = %s', global_step,
                    tr_loss)
    
    tokenizer.save_pretrained(args.output_dir)
    
    # 加载保存的模型和tokenizer
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    model.to(args.device)
    
    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        logger.info('Evaluate the following checkpoints: %s', checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints
                                                           ) > 1 else ''
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + ('_{}'.format(global_step) if global_step else
                                ''), v) for k, v in result.items())
            results.update(result)
    logger.info('Results: {}'.format(results))
    return results


if __name__ == '__main__':
    """
    python -m pdb main.py --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad --do_train --do_eval --do_lower_case --learning_rate 3e-5 --num_train_epochs 10  --per_gpu_eval_batch_size=3 --per_gpu_train_batch_size=8 --device cuda:0 --evaluate_during_training
    """
    main()
