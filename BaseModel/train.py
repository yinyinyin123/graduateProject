import logging 
import random 
import os 
from model import basemodel
from optimization import BertAdam
import pickle
import torch 
import numpy as np
from convertExamplesToFeatures import InputFeatures
from torch.utils.data import DataLoader, RandomSampler,TensorDataset
from tqdm import tqdm

print('import ok')
logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def train(train_batch_size, roberta_model, hidden_size=768, learning_rate=3e-5, warmup_proportion=0.1, seed=23):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    print('loading train features.')
    with open('preprocess/trainFeatures.pkl', 'rb') as f:
        trainFeatures = pickle.load(f)
    print('train features have been loaded.')
    nums = len(trainFeatures)
    print('训练集大小：', nums)
    num_train_optimization_steps = 4 * int(len(trainFeatures) / train_batch_size)
    model = basemodel(roberta_model, hidden_size)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=learning_rate,
                                 warmup=warmup_proportion,
                                 t_total=num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in trainFeatures], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_mask for f in trainFeatures], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in trainFeatures], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in trainFeatures], dtype=torch.long)
    all_answer_choices = torch.tensor([f.ans_choice for f in trainFeatures], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_masks, all_start_positions, all_end_positions, all_answer_choices)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    model.train()
    print('training')
    for epoch in range(4):
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration', disable=False)):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, start_positions, end_positions, answer_choices = batch
            loss = model(input_ids, input_masks, start_positions, end_positions, answer_choices)
            if n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #print(loss)
        #model_to_save = model.module if hasattr(model, 'module') else model
            if(step % (int(nums/train_batch_size)//3) == 0):
                os.mkdir('model'+'/'+str(epoch)+'_'+str(step))
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join('model'+'/'+str(epoch)+'_'+str(step), 'pytorch_model.bin')
                output_config_file = os.path.join('model'+'/'+str(epoch)+'_'+str(step), 'config.json')
                output_m_file = os.path.join('model'+'/'+str(epoch)+'_'+str(step), 'model.pt')
                torch.save(model_to_save.state_dict(),output_m_file)
                torch.save(model_to_save.roberta.state_dict(), output_model_file)
                model_to_save.roberta.config.to_json_file(output_config_file)
            #tokenizer.save_vocabulary(args.output_dir+'/'+str(epoc)+'_'+str(step))


train(20, 'roberta-base')

