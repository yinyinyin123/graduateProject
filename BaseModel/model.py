import torch 
import torch.nn as nn 
from transformers import RobertaModel

class basemodel(nn.Module):

    def __init__(self, roberta_model, hidden_size, answerTypes=4, max_seq_length=512):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.roberta.resize_token_embeddings(50267)
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.get_answer_choice = nn.Linear(hidden_size, answerTypes)
        self.max_seq_length = max_seq_length
        self.lossfn = nn.CrossEntropyLoss(ignore_index=max_seq_length)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, ans_choices=None):
        tokens_output, cls_output = self.roberta(input_ids, attention_mask)
        logits = self.qa_outputs(tokens_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        as_choice_logits = self.get_answer_choice(cls_output)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(ans_choices.size()) > 1:
                ans_choices = ans_choices.squeeze(-1)
            start_positions.clamp_(0, self.max_seq_length)
            end_positions.clamp_(0, self.max_seq_length)
            start_loss= self.lossfn(start_logits, start_positions)
            end_loss = self.lossfn(end_logits, end_positions)
            as_choice_loss = self.lossfn(as_choice_logits, ans_choices)
            total_loss = (as_choice_loss + start_loss + end_loss) / 3
            return total_loss 
        else:
            return start_logits, end_logits, as_choice_logits

    