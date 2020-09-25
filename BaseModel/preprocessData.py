import json
import regex as re
from tqdm import tqdm
import string
from collections import Counter 
import pickle

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def len_preserved_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def len_preserved_space(matchobj):
        return ' ' * len(matchobj.group(0))

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', len_preserved_space, text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return remove_articles(remove_punc(lower(s)))

def split_with_span(s):
    if s.split() == []:
        return [], []
    else:
        return zip(*[(m.group(0), (m.start(), m.end()-1)) for m in re.finditer(r'\S+', s)])

def free_text_to_span(free_text, full_text):
    if free_text == "unknown":
        return "__NA__", -1, -1
    if normalize_answer(free_text) == "yes":
        return "__YES__", -1, -1
    if normalize_answer(free_text) == "no":
        return "__NO__", -1, -1

    free_ls = len_preserved_normalize_answer(free_text).split()
    full_ls, full_span = split_with_span(len_preserved_normalize_answer(full_text))
    if full_ls == []:
        return full_text, 0, len(full_text)

    max_f1, best_index = 0.0, (0, len(full_ls)-1)
    free_cnt = Counter(free_ls)
    for i in range(len(full_ls)):
        full_cnt = Counter()
        for j in range(len(full_ls)):
            if i+j >= len(full_ls): break
            full_cnt[full_ls[i+j]] += 1

            common = free_cnt & full_cnt
            num_same = sum(common.values())
            if num_same == 0: continue
            precision = 1.0 * num_same / (j + 1)
            recall = 1.0 * num_same / len(free_ls)
            f1 = (2 * precision * recall) / (precision + recall)
            if max_f1 < f1:
                max_f1 = f1
                best_index = (i, j)

    assert(best_index is not None)
    (best_i, best_j) = best_index
    char_i, char_j = full_span[best_i][0], full_span[best_i+best_j][1]+1

    return full_text[char_i:char_j], char_i, char_j

class CoQAExample(object):
	"""
    A single example for the CoQA dataset.
	"""
	def __init__(self, questionId, questionText, documentTokens, answerText, answerType, spanStart, spanEnd, dialogId):
		self.questionId = questionId          
		self.questionText = questionText      
		self.documentTokens = documentTokens     
		self.answerText = answerText          
		self.answerType = answerType          
		self.spanStart = spanStart            
		self.spanEnd = spanEnd                
		self.dialogId = dialogId              

def isWhiteSpace(char):
	if(char == ' ' or char == '\t' or char == '\r' or char == '\n' or ord(char) == 0x202F ):
		return True  
	return False

def readCoQAExamples(file, istraining=True, max_question_len=50, pre_turn=2):
    with open(file, 'r', encoding='utf-8') as reader:
        data = json.load(reader)['data']
    examples = []
    dialogId = 0
    cnt = 0
    count_example = 0
    for idx in tqdm(range(len(data))):
        datum = data[idx]
        dialogId += 1
        context = datum['story']
        documentTokens = []
        char_to_word_offset = []
        prev_is_whitespace = True 
        for c in context:
            if isWhiteSpace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    documentTokens.append(c)
                else:
                    documentTokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(documentTokens)-1)
        for qa_idx, q in enumerate(datum['questions']):
            count_example += 1
            qas_id = datum['id'] + '#' + str(q["turn_id"])
            question_text = ''
            temp = qa_idx
            while(qa_idx - temp < pre_turn):
                temp -= 1
                if(temp >= 0):
                    question_text = '[Q] '+datum['questions'][temp]['input_text']+' [A] '+datum['answers'][temp]['input_text']+' '+question_text
                else:
                    break
            question_text = question_text + '[Q] '+ q['input_text']
            q_tokens = question_text.split()
            if(len(q_tokens) > max_question_len):
                cnt += 1
                question_text = ' '.join(q_tokens[-max_question_len:])
            start_position = None
            end_position = None 
            orig_answer_text = None 
            ans_choice = None
            if istraining:
                answer_text = datum['answers'][qa_idx]['input_text']
                span_text = datum['answers'][qa_idx]['span_text']
                orig_answer_text, char_i, char_j = free_text_to_span(answer_text, span_text)
                ans_choice = 0 if orig_answer_text == '__NA__'	else \
                                1 if orig_answer_text == '__YES__'	else \
                                2 if orig_answer_text == '__NO__'	else \
                                3
                if(ans_choice == 3):
                    answer_offset = datum['answers'][qa_idx]['span_start']+char_i
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset+answer_length-1]
                    actual_text = ' '.join(documentTokens[start_position:(end_position+1)])
                    cleaned_answer_text = ' '.join(whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        continue 
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ''
            example = CoQAExample(questionId = qas_id, questionText = question_text,documentTokens = documentTokens, \
            	      answerText = orig_answer_text, answerType = ans_choice, spanStart = start_position, \
            	      spanEnd = end_position, dialogId = dialogId)
            examples.append(example)
    if(istraining):
        print('最大问题长度为'+str(max_question_len)+', 在训练集中超过最大问题长度的占比：', cnt/count_example)
    else:
        print('最大问题长度为'+str(max_question_len)+', 在验证集中超过最大问题长度的占比：', cnt/count_example)
    return examples

def writeData(outputfile, data):
    with open(outputfile, 'wb') as f:
        pickle.dump(data, f)

def test():
    dev = readCoQAExamples('data/coqa-dev-v1.0.json', istraining=False)
    train = readCoQAExamples('data/coqa-train-v1.0.json')
    writeData('preprocess/devExamples.pkl', dev)
    writeData('preprocess/trainExamples.pkl', train)


    



        
