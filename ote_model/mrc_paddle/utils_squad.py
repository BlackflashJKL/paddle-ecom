import paddle
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import math
import collections
from io import open
import jieba
import tqdm
from termcolor import colored
from transformers import BasicTokenizer
logger = logging.getLogger(__name__)


def data_preprocess(data, opinion_level):
    insts = []
    for passage in data:
        event = passage['Descriptor']['text'].strip()
        contents = passage['Doc']['content']
        sents = [content[0].strip() for content in contents]
        labels = [content[1].strip() for content in contents]
        aspects = [content[2].strip() for content in contents]
        if opinion_level == 'sent':
            for idx, label in enumerate(labels):
                if label == 'B' or label == 'I':
                    insts.append([sents[idx].strip(), event, aspects[idx]])
        elif opinion_level == 'segment':
            opinion = ''
            for idx, label in enumerate(labels):
                if label == 'B' and not opinion or label == 'I':
                    opinion += sents[idx]
                elif label == 'B' and opinion:
                    old_aspect = aspects[idx - 1]
                    insts.append([opinion, event, old_aspect])
                    opinion = sents[idx]
                elif label == 'O' and opinion:
                    old_aspect = aspects[idx - 1]
                    insts.append([opinion, event, old_aspect])
                    opinion = ''
            if opinion:
                insts.append([opinion, event, aspects[-1]])
    return insts


def get_boundary(event, aspect, language='english'):
    if language == 'chinese':
        start_idx = -1
        end_idx = -1
        candidate_start_idxs = [idx for idx, word in enumerate(event) if 
            aspect[0] in word]
        candidate_end_idxs = [idx for idx, word in enumerate(event) if 
            aspect[-1] in word]
        spans = [[s_idx, e_idx] for s_idx in candidate_start_idxs for e_idx in
            candidate_end_idxs if s_idx <= e_idx]
        spans.sort(key=lambda x: x[1] - x[0])
        for span in spans:
            if aspect in ''.join(event[span[0]:span[1] + 1]):
                start_idx, end_idx = span
    elif language == 'english':
        start_idx = -1
        end_idx = -1
        aspect = aspect.split()
        candidate_start_idxs = [idx for idx, word in enumerate(event) if 
            aspect[0] in word]
        candidate_end_idxs = [idx for idx, word in enumerate(event) if 
            aspect[-1] in word]
        spans = [[s_idx, e_idx] for s_idx in candidate_start_idxs for e_idx in
            candidate_end_idxs if s_idx <= e_idx]
        spans.sort(key=lambda x: x[1] - x[0])
        for span in spans:
            if ' '.join(aspect) in ' '.join(event[span[0]:span[1] + 1]):
                start_idx, end_idx = span
    if start_idx == -1 and end_idx == -1:
        print(event, aspect)
    return start_idx, end_idx


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self, qas_id, question_text, doc_tokens, orig_answer_text=\
        None, start_position=None, end_position=None, is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ''
        s += 'qas_id: %s' % self.qas_id
        s += ', question_text: %s' % self.question_text
        s += ', doc_tokens: [%s]' % ' '.join(self.doc_tokens)
        if self.start_position:
            s += ', start_position: %d' % self.start_position
        if self.end_position:
            s += ', end_position: %d' % self.end_position
        if self.is_impossible:
            s += ', is_impossible: %r' % self.is_impossible
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, example_index, doc_span_index, tokens,
        token_to_orig_map, token_is_max_context, input_ids, input_mask,
        segment_ids, cls_index, p_mask, paragraph_len, start_position=None,
        end_position=None, is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, language='english', opinion_level='segment'
    ):
    """Read a json file into a list of SquadExample."""
    with open(input_file + '.doc.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)
    try:
        with open(input_file + '.ann.json', 'r', encoding='utf-8') as f:
            opinions = json.load(f)
    except FileNotFoundError:
        opinions = []
        print(colored('[There is no ' + input_file + '.ann.json.]', 'red'))
    opinions_reformat = []
    warning = False
    for doc in docs:
        event = doc['Descriptor']['text']
        event_id = doc['Descriptor']['event_id']
        doc_id = doc['Doc']['doc_id']
        contents = doc['Doc']['content']
        sents = [content['sent_text'] for content in contents]
        doc_opinions = [opinion for opinion in opinions if int(opinion[
            'doc_id']) == int(doc_id)]
        for doc_opinion in doc_opinions:
            opinion_text = sents[doc_opinion['start_sent_idx']:doc_opinion[
                'end_sent_idx'] + 1]
            try:
                opinions_reformat.append([opinion_text, event, doc_opinion[
                    'argument'], event_id, doc_id, doc_opinion[
                    'start_sent_idx'], doc_opinion['end_sent_idx']])
            except KeyError:
                if not warning:
                    print(colored('[There is no gold argument!]', 'red'))
                    warning = True
                opinions_reformat.append([opinion_text, event, '', event_id,
                    doc_id, doc_opinion['start_sent_idx'], doc_opinion[
                    'end_sent_idx']])
    examples = []
    for idx, opinion in enumerate(opinions_reformat):
        qas_id = idx
        question_text = ' '.join(opinion[0])
        is_impossible = False
        if language == 'english':
            doc_tokens = opinion[1].strip().split()
        elif language == 'chinese':
            doc_tokens = list(jieba.cut(opinion[1].strip()))
        if not is_impossible and len(opinion[2]):
            answer = opinion[2]
            orig_answer_text = opinion[2]
            start_position, end_position = get_boundary(doc_tokens, answer,
                language)
        else:
            start_position = -1
            end_position = -1
            orig_answer_text = ''
        example = SquadExample(qas_id=qas_id, question_text=question_text,
            doc_tokens=doc_tokens, orig_answer_text=orig_answer_text,
            start_position=start_position, end_position=end_position,
            is_impossible=is_impossible)
        example.event_id = opinion[3]
        example.doc_id = opinion[4]
        example.start_sent_idx = opinion[5]
        example.end_sent_idx = opinion[6]
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
    doc_stride, max_query_length, is_training, cls_token_at_end=False,
    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
    sequence_a_segment_id=0, sequence_b_segment_id=1, cls_token_segment_id=\
    0, pad_token_segment_id=0, mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""
    unique_id = 1000000000
    features = []
    for example_index, example in enumerate(tqdm.tqdm(examples)):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for i, token in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1
                    ] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            tok_start_position, tok_end_position = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                tokenizer, example.orig_answer_text)
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        _DocSpan = collections.namedtuple('DocSpan', ['start', 'length'])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        for doc_span_index, doc_span in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            p_mask = []
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[
                    split_token_index]
                is_max_context = _check_is_max_context(doc_spans,
                    doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and 
                    tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = (tok_start_position - doc_start +
                        doc_offset)
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index
            features.append(InputFeatures(unique_id=unique_id,
                example_index=example_index, doc_span_index=doc_span_index,
                tokens=tokens, token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context, input_ids=\
                input_ids, input_mask=input_mask, segment_ids=segment_ids,
                cls_index=cls_index, p_mask=p_mask, paragraph_len=\
                paragraph_len, start_position=start_position, end_position=\
                end_position, is_impossible=span_is_impossible))
            unique_id += 1
    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
    orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = ' '.join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = ' '.join(doc_tokens[new_start:new_end + 1])
            if text_span == tok_answer_text:
                return new_start, new_end
    return input_start, input_end


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context
            ) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index


RawResult = collections.namedtuple('RawResult', ['unique_id',
    'start_logits', 'end_logits'])


def write_predictions(all_examples, all_features, all_results, n_best_size,
    max_answer_length, do_lower_case, output_prediction_file,
    output_nbest_file, output_null_log_odds_file, verbose_logging,
    version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info('Writing predictions to: %s' % output_prediction_file)
    logger.info('Writing nbest to: %s' % output_nbest_file)
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple('PrelimPrediction', [
        'feature_index', 'start_index', 'end_index', 'start_logit',
        'end_logit'])
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    for example_index, example in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        score_null = 1000000
        min_null_feature_index = 0
        null_start_logit = 0
        null_end_logit = 0
        for feature_index, feature in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            if version_2_with_negative:
                feature_null_score = result.start_logits[0
                    ] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False
                        ):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(_PrelimPrediction(
                        feature_index=feature_index, start_index=\
                        start_index, end_index=end_index, start_logit=\
                        result.start_logits[start_index], end_logit=result.
                        end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(_PrelimPrediction(feature_index=\
                min_null_feature_index, start_index=0, end_index=0,
                start_logit=null_start_logit, end_logit=null_end_logit))
        prelim_predictions = sorted(prelim_predictions, key=lambda x: x.
            start_logit + x.end_logit, reverse=True)
        _NbestPrediction = collections.namedtuple('NbestPrediction', [
            'text', 'start_logit', 'end_logit'])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:pred.end_index + 1
                    ]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:
                    orig_doc_end + 1]
                tok_text = ' '.join(tok_tokens)
                tok_text = tok_text.replace(' ##', '')
                tok_text = tok_text.replace('##', '')
                tok_text = tok_text.strip()
                tok_text = ' '.join(tok_text.split())
                orig_text = ' '.join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text,
                    do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ''
                seen_predictions[final_text] = True
            nbest.append(_NbestPrediction(text=final_text, start_logit=pred
                .start_logit, end_logit=pred.end_logit))
        if version_2_with_negative:
            if '' not in seen_predictions:
                nbest.append(_NbestPrediction(text='', start_logit=\
                    null_start_logit, end_logit=null_end_logit))
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text='empty', start_logit=\
                    0.0, end_logit=0.0))
        if not nbest:
            nbest.append(_NbestPrediction(text='empty', start_logit=0.0,
                end_logit=0.0))
        assert len(nbest) >= 1
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
        probs = _compute_softmax(total_scores)
        nbest_json = []
        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = entry.text
            output['probability'] = probs[i]
            output['start_logit'] = entry.start_logit
            output['end_logit'] = entry.end_logit
            nbest_json.append(output)
        assert len(nbest_json) >= 1
        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]['text']
        else:
            score_diff = (score_null - best_non_null_entry.start_logit -
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ''
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json
    with open(output_prediction_file, 'w') as writer:
        writer.write(json.dumps(all_predictions, indent=4) + '\n')
    with open(output_nbest_file, 'w') as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
    if version_2_with_negative:
        with open(output_null_log_odds_file, 'w') as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + '\n')
    return all_predictions


RawResultExtended = collections.namedtuple('RawResultExtended', [
    'unique_id', 'start_top_log_probs', 'start_top_index',
    'end_top_log_probs', 'end_top_index', 'cls_logits'])


def write_predictions_extended(all_examples, all_features, all_results,
    n_best_size, max_answer_length, output_prediction_file,
    output_nbest_file, output_null_log_odds_file, orig_data_file,
    start_n_top, end_n_top, version_2_with_negative, tokenizer, verbose_logging
    ):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.
        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple('PrelimPrediction', [
        'feature_index', 'start_index', 'end_index', 'start_log_prob',
        'end_log_prob'])
    _NbestPrediction = collections.namedtuple('NbestPrediction', ['text',
        'start_log_prob', 'end_log_prob'])
    logger.info('Writing predictions to: %s', output_prediction_file)
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    for example_index, example in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        score_null = 1000000
        for feature_index, feature in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            cur_null_score = result.cls_logits
            score_null = min(score_null, cur_null_score)
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]
                    j_index = i * end_n_top + j
                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue
                    if not feature.token_is_max_context.get(start_index, False
                        ):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(_PrelimPrediction(
                        feature_index=feature_index, start_index=\
                        start_index, end_index=end_index, start_log_prob=\
                        start_log_prob, end_log_prob=end_log_prob))
        prelim_predictions = sorted(prelim_predictions, key=lambda x: x.
            start_log_prob + x.end_log_prob, reverse=True)
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            tok_tokens = feature.tokens[pred.start_index:pred.end_index + 1]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:orig_doc_end + 1]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
            tok_text = tok_text.strip()
            tok_text = ' '.join(tok_text.split())
            orig_text = ' '.join(orig_tokens)
            final_text = get_final_text(tok_text, orig_text, tokenizer.
                do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True
            nbest.append(_NbestPrediction(text=final_text, start_log_prob=\
                pred.start_log_prob, end_log_prob=pred.end_log_prob))
        if not nbest:
            nbest.append(_NbestPrediction(text='', start_log_prob=-
                1000000.0, end_log_prob=-1000000.0))
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry
        probs = _compute_softmax(total_scores)
        nbest_json = []
        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = entry.text
            output['probability'] = probs[i]
            output['start_log_prob'] = entry.start_log_prob
            output['end_log_prob'] = entry.end_log_prob
            nbest_json.append(output)
        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None
        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json
    with open(output_prediction_file, 'w') as writer:
        writer.write(json.dumps(all_predictions, indent=4) + '\n')
    with open(output_nbest_file, 'w') as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
    if version_2_with_negative:
        with open(output_null_log_odds_file, 'w') as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + '\n')
    with open(orig_data_file, 'r', encoding='utf-8') as reader:
        orig_data = json.load(reader)['data']
    out_eval = {}
    return out_eval


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for i, c in enumerate(text):
            if c == ' ':
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = ''.join(ns_chars)
        return ns_text, ns_to_s_map
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = ' '.join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text,
                orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1
    orig_ns_text, orig_ns_to_s_map = _strip_spaces(orig_text)
    tok_ns_text, tok_ns_to_s_map = _strip_spaces(tok_text)
    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'"
                , orig_ns_text, tok_ns_text)
        return orig_text
    tok_s_to_ns_map = {}
    for i, tok_index in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i
    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]
    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text
    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]
    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text
    output_text = orig_text[orig_start_position:orig_end_position + 1]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse
        =True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []
    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score
    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x
    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
