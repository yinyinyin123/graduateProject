from transformers import RobertaTokenizer
from preprocessData import *
import pickle
import collections
import logging 
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 start_position=None,
                 end_position=None,
                 ans_choice=None,
                 dialog_id=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.start_position = start_position
        self.end_position = end_position
        self.ans_choice = ans_choice
        self.dialog_id = dialog_id

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index

def convert_examples_to_features(examples, tokenizer, max_seq_length=512, doc_stride=128, max_query_length=60, isTraining=True):
    unique_id = 100000000
    features = []
    for idx in tqdm(range(len(examples))):
        example = examples[idx]
        query_tokens = tokenizer.tokenize(example.questionText)
        if(len(query_tokens) > max_query_length):
            query_tokens = query_tokens[-max_query_length:]
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.documentTokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        tok_start_position = None 
        tok_end_position = None 
        if isTraining and example.answerType != 3:
            tok_start_position = -1
            tok_end_position = -1
        if isTraining and example.answerType == 3:
            tok_start_position = orig_to_tok_index[example.spanStart]
            if example.spanEnd < len(example.documentTokens) - 1:
                tok_end_position = orig_to_tok_index[example.spanEnd + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.answerText)
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        _DocSpan = collections.namedtuple(  
            "DocSpan", ["start", "length"])
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
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            tokens.append('<s>')
            for token in query_tokens:
                tokens.append(token)
            tokens.append('</s>')
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i 
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context 
                tokens.append(all_doc_tokens[split_token_index])
            tokens.append('</s>')
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            while(len(input_ids) < max_seq_length):
                input_ids.append(1)
                input_mask.append(0)
            start_position = None
            end_position = None 
            if isTraining and example.answerType == 3:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False 
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True 
                if out_of_span:
                    start_position = -100
                    end_position = -100
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset 
                    end_position = tok_end_position - doc_start + doc_offset 
            if isTraining and example.answerType != 3:
                start_position = -100
                end_position = -100
            features.append(InputFeatures(
                unique_id = unique_id, 
                example_index = idx, 
                doc_span_index = doc_span_index, 
                tokens = tokens, 
                token_to_orig_map = token_to_orig_map, 
                token_is_max_context = token_is_max_context, 
                input_ids = input_ids,
                input_mask = input_mask, 
                start_position = start_position, 
                end_position = end_position, 
                ans_choice = example.answerType, 
                dialog_id = example.dialogId
            ))
            unique_id += 1
    return features

def test():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    special_tokens_dict = {'additional_special_tokens': ["[Q]", "[A]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    with open('preprocess/trainExamples.pkl', 'rb') as f:
        examples = pickle.load(f)
    print('convert train examples to features')
    features = convert_examples_to_features(examples, tokenizer)
    print('train examples converted and saving....')
    writeData('preprocess/trainFeatures.pkl', features)
    print('train features saved.')
    with open('preprocess/devExamples.pkl', 'rb') as f:
        examples = pickle.load(f)
    print('convert dev examples to features')
    features = convert_examples_to_features(examples, tokenizer)
    print('dev examples converted and saving ')
    writeData('preprocess/devFeatures.pkl', features)
    print('train features saved')

#test()