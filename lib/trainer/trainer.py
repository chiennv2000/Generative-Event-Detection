import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import get_linear_schedule_with_warmup, AdamW
from transformers.generation_logits_process import LogitsProcessorList, TopKLogitsWarper
from transformers.generation_stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer

import lib

class MT5Trainer(pl.LightningModule):
    def __init__(self, tokenizer, args):
        super().__init__()
        
        self.args = args
        self.model = lib.ConstraintMT5ForConditionalGeneration.from_pretrained(self.args.pretrained_path)
        
        event_tokens = lib.data.utils.get_event_tokens(tokenizer)
        setattr(self.model, "event_tokens", event_tokens)
        self.tokenizer = tokenizer
        
        self.CLEvaluator = lib.CrossLingualEvaluator(tokenizer, args)
        self.RewardScorers = lib.RewardScorer(args.muse_src_lang_path, args.muse_tgt_lang_path)
        
    def forward(self,
                source_ids,
                source_attention_mask=None,
                target_attention_mask=None,
                labels=None):
        
        return self.model(input_ids=source_ids, 
                          attention_mask=source_attention_mask,
                          decoder_attention_mask=target_attention_mask,
                          labels=labels)
        
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        self.args.n_steps = len(self.train_dataloader())//self.args.accumulate_grad_batches
        self.scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                         num_training_steps=self.args.n_epochs*self.args.n_steps,
                                                         num_warmup_steps=0)
        
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False,using_lbfgs=False):
        if (batch_idx + 1) % self.args.accumulate_grad_batches == 0:
            optimizer.step(closure=optimizer_closure)
            self.scheduler.step()
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        if (batch_idx + 1) % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()
            
    def generate_from_batch(self, batch, num_beams=3, max_length=190):
        predicted_texts = []
        with torch.no_grad():
            output_ids = self.model.generate(batch["source_ids"], num_beams=num_beams, max_length=max_length)
            predicted_texts.extend(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        return predicted_texts
    
    def compute_MLE_loss(self,
                         source_ids,
                         source_attention_mask,
                         target_ids,
                         target_attention_mask
                         ):
        labels = target_ids.clone().detach()
        labels[target_ids == self.tokenizer.pad_token_id] = -100
        
        outputs = self(source_ids=source_ids,
                       source_attention_mask=source_attention_mask,
                       target_attention_mask=target_attention_mask,
                       labels=labels)
        
        return outputs[0]
    
    def compute_reward(self, aug_target_ids, target_ids):
        aug_target_texts = self.tokenizer.batch_decode(aug_target_ids, skip_special_tokens=True)
        target_texts = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
        
        aug_events = [re.findall('\[(.*?)\]', text) for text in aug_target_texts]
        actual_events = [re.findall('\[(.*?)\]', text) for text in target_texts]
        R1, R2, R3 = self.RewardScorers.compute_rewards(aug_events, actual_events)
        
        rewards = torch.FloatTensor(R1) + 0.1*torch.FloatTensor(R2) + 0.01*torch.FloatTensor(R3)

        return rewards.to(self.model.device)
    
    def prepare_decoder_input_ids_for_generation(self,
                                                 input_ids: torch.LongTensor,
                                                 decoder_start_token_id: int = 0):
        decoder_input_ids = (torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device) * decoder_start_token_id)
        return decoder_input_ids

    def compute_RL_loss(self,
                        aug_source_ids,
                        aug_source_attention_mask,
                        target_ids):
        encoder_outputs = self.model.encoder(input_ids=aug_source_ids, attention_mask=aug_source_attention_mask,output_attentions=True, output_hidden_states=True, return_dict=True)
        decoder_input_ids = self.prepare_decoder_input_ids_for_generation(aug_source_ids, decoder_start_token_id=0)
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=self.args.tgt_max_length))
        
        with torch.no_grad():
            baselines = self.model.greedy_search(input_ids=decoder_input_ids,
                                                 pad_token_id=0, eos_token_id=1,
                                                 encoder_outputs=encoder_outputs,
                                                 stopping_criteria=stopping_criteria,
                                                 attention_mask=aug_source_attention_mask)
            b_rewards = self.compute_reward(baselines, target_ids)
        
        # warpers = LogitsProcessorList()
        # warpers.append(TopKLogitsWarper(top_k=25000, min_tokens_to_keep=1))
        
        # sample_outputs = self.model.sample(input_ids=decoder_input_ids,
        #                                    # logits_warper=warpers,
        #                                    pad_token_id=0,
        #                                    eos_token_id=1,
        #                                    encoder_outputs=encoder_outputs,
        #                                    stopping_criteria=stopping_criteria,
        #                                    output_scores=True,
        #                                    return_dict_in_generate=True,
        #                                    attention_mask=aug_source_attention_mask)
        
        # sample_rewards = self.compute_reward(sample_outputs.sequences, target_ids)

        # sequences = sample_outputs.sequences[:, 1:]

        # scores = torch.stack(sample_outputs.scores)
        # scores = F.softmax(scores, dim=-1)
        # scores = torch.transpose(scores, 0, 1)
        #num_beams = 2
        beam_scorer = BeamSearchScorer(batch_size=aug_source_ids.shape[0],
                                       num_beams=self.args.num_beams,
                                       device=self.model.device,
                                       length_penalty=1.0)
        
        decoder_input_ids, _ = self.model._expand_inputs_for_generation(decoder_input_ids, expand_size=self.args.num_beams, is_encoder_decoder=True, encoder_outputs=encoder_outputs)
        
        sample_outputs = self.model.beam_search(input_ids=decoder_input_ids,
                                         encoder_outputs=encoder_outputs,
                                         pad_token_id=0,
                                         eos_token_id=1,
                                         beam_scorer=beam_scorer,
                                         return_dict_in_generate=True,
                                         stopping_criteria=stopping_criteria,
                                         output_scores=True)
        
        sample_rewards = self.compute_reward(sample_outputs.sequences, target_ids)
        
        sequences = sample_outputs.sequences[:, 1:]
        
        scores = torch.stack(sample_outputs.scores)

        scores = F.softmax(scores, dim=-1)
        scores = torch.transpose(scores, 0, 1)
        scores = scores[[i for i in range(scores.size(0)) if not i%self.args.num_beams], :sequences.size(1), :]
        
        RL_log_probs = []
        for i in range(scores.size(0)):
            RL_log_probs.append(F.nll_loss(scores[i], sequences[i], ignore_index=self.tokenizer.pad_token_id, reduction="sum"))
        RL_log_probs = torch.stack(RL_log_probs)
        RL_loss = - (sample_rewards - b_rewards) * RL_log_probs
        
        del encoder_outputs
        del sample_outputs
        del decoder_input_ids
        # del warpers
        
        return torch.mean(RL_loss), torch.mean(sample_rewards).item() 
          
    
    def basic_step(self, batch, is_training=True):
        MLE_loss = self.compute_MLE_loss(batch["source_ids"],
                                         batch["source_attention_mask"],
                                         batch["target_ids"],
                                         batch["target_attention_mask"])
        if is_training:
            # RL_loss, rewards = self.compute_RL_loss(batch['aug_source_ids'],
            #                                         batch['aug_source_attention_mask'],
            #                                         batch['target_ids'])
            
            # aug_MLE_loss = self.compute_MLE_loss(batch["aug_source_ids"],
            #                                      batch["aug_source_attention_mask"],
            #                                      batch["aug_target_ids"],
            #                                      batch["aug_target_attention_mask"])
            # loss = RL_loss + MLE_loss + aug_MLE_loss
            # return loss, rewards

            RL_loss, rewards = self.compute_RL_loss(batch['source_ids'],
                                                    batch['source_ids'],
                                                    batch['target_ids'])
            loss = RL_loss + MLE_loss
            return loss, rewards

        else:
            loss = MLE_loss
            gold_triggers, pred_triggers = [], []
            gold_events, pred_events = [], []
            predicted_labels = self.generate_from_batch(batch, num_beams=3, max_length=self.args.tgt_max_length)

            for i in range(batch["source_ids"].size(0)):
                actual_label = self.tokenizer.decode(batch["target_ids"][i], skip_special_tokens=True)

                pred_sent_triggers, pred_sent_events = lib.Evaluator.parse(predicted_labels[i])
                pred_triggers.append(pred_sent_triggers)
                pred_events.append(pred_sent_events)

                gold_sent_triggers, gold_sent_events = lib.Evaluator.parse(actual_label)
                gold_triggers.append(gold_sent_triggers)
                gold_events.append(gold_sent_events)
            
            return loss, gold_events, pred_events
    
    def training_step(self, train_batch, batch_idx):
        loss, rewards = self.basic_step(train_batch, is_training=True)
        self.log("train_loss", loss, logger=False)
        self.log("reward", rewards, prog_bar=True)
        return loss
        
        # loss = self.basic_step(train_batch, is_training=True)
        # self.log("train_loss", loss, logger=False)
        # return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss, gold_events, pred_events = self.basic_step(val_batch, is_training=False)
        return {
            "val_loss_step": loss,
            "gold_events": gold_events,
            "pred_events": pred_events,
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        val_loss_mean = 0
        gold_events, pred_events = [], []
        for out in validation_step_outputs:
            val_loss_mean += out["val_loss_step"]
            gold_events.extend(out["gold_events"])
            pred_events.extend(out["pred_events"])
        
        val_loss_mean /= len(validation_step_outputs)
        f1_class = lib.Evaluator.calc_f1(pred_events, gold_events)

        evaluation = {"val_loss_mean": val_loss_mean,
                      "f1_class": f1_class}

        result = {"log": evaluation,
                  "val_loss_mean": val_loss_mean,
                  "f1_class": f1_class}
        
        self.log("val_loss", val_loss_mean, logger=False, on_epoch=True)
        self.log("f1_class", f1_class, prog_bar=True, on_epoch=True)
        
        if f1_class >= 0.6:
            self.CLEvaluator.evaluate_cross_lingual(self.current_epoch, self.model, val_loss_mean.item(), f1_class, langs=["zh", "ar"], file_name=self.args.output_log)
        
        return result
    
    def train_dataloader(self):
        train_data = lib.data.AnnotationACE05(self.args.train_data_path, self.tokenizer, aug_lang=self.args.aug_lang, is_training=True)
        print(train_data[7])
        dataloader = DataLoader(train_data, batch_size=self.args.train_batch_size, shuffle=True, num_workers=4, collate_fn=train_data.collate_fn)
        return dataloader
    
    def val_dataloader(self):
        val_data = lib.data.AnnotationACE05(self.args.val_data_path, self.tokenizer, is_training=False)
        val_dataloader = DataLoader(val_data, batch_size=self.args.val_batch_size, shuffle=False, num_workers=4, collate_fn=val_data.collate_fn)
        return val_dataloader
        