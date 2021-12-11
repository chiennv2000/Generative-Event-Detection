import json
from re import M
from torch.utils.data import Dataset

class AnnotationACE05(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 aug_lang='zh',
                 is_training=False):
        
        self.tokenizer = tokenizer
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.src_texts = data['src_texts']
        self.tgt_texts = data['tgt_texts']
        if is_training:
            if aug_lang == "en":
                self.aug_src_texts = data['src_en_texts']
                self.aug_tgt_texts = data['tgt_en_texts']

            if aug_lang == "zh":
                self.aug_src_texts = data['src_zh_texts']
                self.aug_tgt_texts = data['tgt_zh_texts']
            
            if aug_lang == "ar":
                self.aug_src_texts = data['src_ar_texts']
                self.aug_tgt_texts = data['tgt_ar_texts']
        else:
            self.aug_src_texts = [""]*len(data['src_texts'])
            self.aug_tgt_texts = [""]*len(data['tgt_texts'])
          
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        return {'src_text': self.src_texts[idx],
                'tgt_text': self.tgt_texts[idx],
                'aug_src_text': self.aug_src_texts[idx],
                'aug_tgt_text': self.aug_tgt_texts[idx]}
    
    def collate_fn(self, batch):
        src_texts = [item['src_text'] for item in batch]
        tgt_texts = [item['tgt_text'] for item in batch]
        
        aug_src_texts = [item['aug_src_text'] for item in batch]
        aug_tgt_texts = [item['aug_tgt_text'] for item in batch]
        
        # src_encodings = self.tokenizer.batch_encode_plus(src_texts,
        #                                                  padding=True,
        #                                                  return_tensors='pt',
        #                                                  add_special_tokens=True)
        
        # tgt_encodings = self.tokenizer.batch_encode_plus(tgt_texts,
        #                                                  padding=True,
        #                                                  return_tensors='pt',
        #                                                  add_special_tokens=True)
        
        # aug_src_encodings = self.tokenizer.batch_encode_plus(aug_src_texts,
        #                                                      padding=True,
        #                                                      return_tensors='pt',
        #                                                      add_special_tokens=True)
        
        # aug_tgt_encodings = self.tokenizer.batch_encode_plus(aug_tgt_texts,
        #                                                      padding=True,
        #                                                      return_tensors='pt',
        #                                                      add_special_tokens=True)
        
        src_encodings = self.tokenizer.batch_encode_plus(src_texts,
                                                         truncation=True,
                                                         padding=True,
                                                         return_tensors='pt',
                                                         max_length=160,
                                                         add_special_tokens=True)
        
        tgt_encodings = self.tokenizer.batch_encode_plus(tgt_texts,
                                                         truncation=True,
                                                         padding=True,
                                                         return_tensors='pt',
                                                         max_length=160,
                                                         add_special_tokens=True)
        
        aug_src_encodings = self.tokenizer.batch_encode_plus(aug_src_texts,
                                                             truncation=True,
                                                             padding=True,
                                                             return_tensors='pt',
                                                             max_length=160,
                                                             add_special_tokens=True)
        
        aug_tgt_encodings = self.tokenizer.batch_encode_plus(aug_tgt_texts,
                                                             truncation=True,
                                                             padding=True,
                                                             return_tensors='pt',
                                                             max_length=160,
                                                             add_special_tokens=True)
        
        return {'source_ids': src_encodings['input_ids'],
                'source_attention_mask': src_encodings['attention_mask'],
                'target_ids': tgt_encodings['input_ids'],
                'target_attention_mask': tgt_encodings['attention_mask'],
                'aug_source_ids': aug_src_encodings['input_ids'],
                'aug_source_attention_mask': aug_src_encodings['attention_mask'],
                'aug_target_ids': aug_tgt_encodings['input_ids'],
                'aug_target_attention_mask': aug_tgt_encodings['attention_mask']}
