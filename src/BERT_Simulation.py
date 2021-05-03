from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

class BERT:
    def __init__(self):
        self.output = "../"
    
    def load_module(self):
        BertForSequenceClassification.from_pretrained(self.output)
    
    def ret_tokenizer(self):
        BertTokenizer.from_pretrained(self.output)
    
    def encondizer(self):
        token = self.ret_tokenizer()
        validation = "" #Extract sentences from CoLa and replace
        return token.encode_plus(validation, max_length=60, pad_to_max_length=True, return_attention_mask=True)

    def simulate(self):
        tokenizer = self.encondizer()
        valid_ID = tokenizer['input_ids']
        valid_MASKS = tokenizer['attention_mask']
        verification = torch.LongTensor(valid_ID)
        return verification, valid_ID, valid_MASKS
    
    def torch_app(self):
        torch_tensor, ID, MASKS = self.simulate()
        model = self.load_module()
        new_input = ID.to("cpu")
        new_mask = MASKS.to("cpu")
        return new_input, new_mask, model
    
    def outputs(self):
        ID, MASK, MODEL = self.torch_app()
        return MODEL(ID, attention_mask=MASK)
