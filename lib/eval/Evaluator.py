from logging import log
import itertools
import os, io, json, re
import lib
import numpy as np
import torch
from torch.utils.data import DataLoader


def parse(text):
    text = text.replace(" ", "")
    indices = [(m.start(0), m.end(0)) for m in re.finditer('\[(.*?)\]', text)]
    shift_index = 0
    triggers, events = [], []
    for start, end in indices:
        start, end = start - shift_index, end - shift_index
        try:
            trigger, event_type = text[start + 1: end - 1].split("|")
        except:
            break
        events.append((start, trigger.strip(), event_type.strip()))
        triggers.append((start, trigger.strip()))

        text = text[0: start] + trigger.strip() + text[end: ]
        shift_index += (end - start) - len(trigger.strip())
    
    return triggers, events

def calc_f1(preds, golds):
    gold_n, pred_n, true_positive_n = 0, 0, 0
    for i in range(len(golds)):
        for e in preds[i]:
            pred_n += 1
        for e in golds[i]:
            gold_n += 1
        for e_pred in preds[i]:
            if e_pred in golds[i]:
                true_positive_n += 1

    precision, recall, f1 = 0, 0, 0
    if pred_n != 0:
        precision = true_positive_n/pred_n
    else:
        pred_n = 1
    if gold_n != 0:
        recall = true_positive_n/gold_n
    else:
        recall = 1
    if precision or recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    return f1
    

def get_scores(preds, golds):
    # y_pred = list(itertools.chain(*preds))
    # y_true = list(itertools.chain(*golds))
    y_pred = preds
    y_true = golds
    
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return f1

class CrossLingualEvaluator(object):
    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        
        self.en_data =  lib.data.AnnotationACE05(args.en_test_data_path, self.tokenizer)
        self.en_loader = DataLoader(self.en_data, batch_size=self.args.val_batch_size, collate_fn=self.en_data.collate_fn, shuffle=False, num_workers=4)
        
        self.zh_data = lib.data.AnnotationACE05(args.zh_test_data_path, self.tokenizer)
        self.zh_loader = DataLoader(self.zh_data, batch_size=self.args.val_batch_size, collate_fn=self.zh_data.collate_fn, shuffle=False, num_workers=4)
        
        self.ar_data = lib.data.AnnotationACE05(args.ar_test_data_path, self.tokenizer)
        self.ar_loader = DataLoader(self.ar_data, batch_size=self.args.val_batch_size, collate_fn=self.ar_data.collate_fn, shuffle=False, num_workers=4)
    
    def evaluate(self, model, data, dataloader, device):
        gold_triggers, pred_triggers = [], []
        gold_events, pred_events = [], []
        
        predicted_texts = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                output_ids = model.generate(batch["source_ids"].to(device), num_beams=3, max_length=self.args.tgt_max_length)
                predicted_texts.extend(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        
        for i in range(len(data)):
            actual_label = data[i]["tgt_text"]

            pred_sent_triggers, pred_sent_events = parse(predicted_texts[i])
            pred_triggers.extend([(i, *trigger) for trigger in pred_sent_triggers])
            pred_events.extend([(i, *event) for event in pred_sent_events])

            gold_sent_triggers, gold_sent_events = parse(actual_label)
            gold_triggers.extend([(i, *trigger) for trigger in gold_sent_triggers])
            gold_events.extend([(i, *event) for event in gold_sent_events])

        f1_indentification = get_scores(pred_triggers, gold_triggers)
        f1_classification = get_scores(pred_events, gold_events)
    
        return f1_indentification, f1_classification
    
    def evaluate_cross_lingual(self, epoch, model, val_loss, f1_class, langs=['zh', 'ar'], file_name=""):
        results = {
            'f1_en': -1,
            'f1_zh': -1,
            'f1_ar': -1,
            'val_loss': val_loss,
            'f1_class_on_dev': f1_class
        }
        
        if 'en' in langs:
            _, f1_en = self.evaluate(model, self.en_data, self.en_loader, model.device)
            results['f1_en'] = f1_en
        if 'zh' in langs:
            _, f1_zh = self.evaluate(model, self.zh_data, self.zh_loader, model.device)
            results['f1_zh'] = f1_zh
        if 'ar' in langs:
            _, f1_ar = self.evaluate(model, self.ar_data, self.ar_loader, model.device)
            results['f1_ar'] = f1_ar
            
        if file_name == "":
            log_path = os.path.join(self.args.log_dir, 'results_rl_zh_to'+ self.args.aug_lang +'.json')
        else:
            log_path = os.path.join(self.args.log_dir, file_name)
            
        if os.path.isfile(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                full_results = json.load(f)
            full_results['epoch_' + str(epoch)] = results
        else:
            full_results = {'epoch_' + str(epoch): results}
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(full_results, indent=4, ensure_ascii=False))

class RewardScorer(object):
    def __init__(self, muse_src_lang_path, muse_tgt_lang_path):
        self.src_embeddings, self.src_id2word, self.src_word2id = self.load_vec(muse_src_lang_path)
        self.tgt_embeddings, self.tgt_id2word, self.tgt_word2id = self.load_vec(muse_tgt_lang_path)

    
    def calc_f1_reward(self, event_preds, event_golds):
        num_proposed = len(event_preds)
        num_gold = len(event_golds)

        y_true_set = set(event_golds)
        num_correct = 0
        for item in event_preds:
            if item in y_true_set:
                num_correct += 1

        if num_proposed != 0:
            precision = num_correct / num_proposed
        else:
            precision = 1.0

        if num_gold != 0:
            recall = num_correct / num_gold
        else:
            recall = 1.0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        return f1
    
    def calc_num_events_reward(self, event_preds, event_golds):
        if len(event_preds) == len(event_golds):
            return 1
        else:
            return 0
    
    def calc_semantic_reward(self, trigger_preds, triggers_golds):
        full_triggers_preds = " ".join(trigger_preds)
        full_triggers_golds = " ".join(triggers_golds)

        v1 = self.get_representation_vector(full_triggers_preds, self.tgt_word2id, self.tgt_embeddings)
        v2 = self.get_representation_vector(full_triggers_golds, self.src_word2id, self.src_embeddings)

        return self.calc_cosine(v1, v2)
    
    def load_vec(self, emb_path, nmax=-1):
        vectors = []
        word2id = {}
        count = 0
        with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                assert word not in word2id, 'word found twice'
                if vect.shape[0] != 300:
                    vect = np.zeros(300)
                    count += 1
                vectors.append(vect)
                word2id[word] = len(word2id)
                if len(word2id) == nmax:
                    break
        id2word = {v: k for k, v in word2id.items()}
        embeddings = np.vstack(vectors)
        print(f"There are {count} vectors with diffirent 300 dim")
        return embeddings, id2word, word2id
    
    def get_representation_vector(self, text, word2id, embeddings):
        embeded_vector = []
        for word in text.split():
            try:
                embeded_vector.append(embeddings[word2id[word]])
            except KeyError:
                continue
        # print(embeded_vector)
        if len(embeded_vector) == 0: 
            return np.zeros(embeddings.shape[1]) + 1e-12
        else:
            return np.stack(embeded_vector, axis=0).sum(axis=0)
        
    def split_triggers_and_events(self, markers):
        """
        markers: [' talks | Contact:Meet ', ' talks | Contact:Meet ']
        Returns:
        triggers = ['talls', 'talks']
        events = ['Contact:Meet', 'Contact:Meet']
        """
        events = []
        triggers = []
        for marker in markers:
            try:
                event_index = marker.index("|")
                events.append(marker[event_index + 1: ].strip())
                triggers.append(marker[: event_index].strip())
            except ValueError:
                continue
        return triggers, events

    def calc_cosine(self, v1, v2):
        return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    
    def compute_rewards(self, preds, golds):
        """
        preds: [[' talks | Contact:Meet ', ' talks | Contact:Meet '], [' visited | Movement:Transport '], [' talks | Contact:Meet ']]\
        golds: [[' 会谈 | Contact:Meet '], [' 访问 | Movement:Transport '], [' 会谈 | Contact:Meet ']]
        
        R1 = [0, 0, 0], R2 = [0, 0, 0], [0, 0, 0]
        """
        assert len(preds) == len(golds)
        R1, R2, R3 = [], [], []
        for pred, gold in zip(preds, golds):
            pred_triggers, pred_events = self.split_triggers_and_events(pred)
            actual_triggers, actual_events = self.split_triggers_and_events(gold)
            
            R1.append(self.calc_f1_reward(pred_events, actual_events))
            if len(pred_triggers) == 0 or len(actual_triggers) == 0:
                R2.append(0)
            else:
                R2.append(self.calc_semantic_reward(pred_triggers, actual_triggers))
            R3.append(self.calc_num_events_reward(pred_triggers, actual_triggers))
            
        return R1, R2, R3
            
        
        
        
        
        
        
    
    
    
    