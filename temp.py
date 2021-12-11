from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import lib
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.file_utils import ModelOutput
import torch.nn.functional as F
import torch

tokenizer = MT5Tokenizer.from_pretrained("pretrained-models/en-zh-epoch=08-f1_class=0.7052.ckpt")
model = MT5ForConditionalGeneration.from_pretrained("pretrained-models/en-zh-epoch=08-f1_class=0.7052.ckpt")
# event_tokens = lib.data.utils.get_event_tokens(tokenizer)
# setattr(model, "event_tokens", event_tokens)

# text = 'Mohamed Hosni Mubarak, in addition to the dialogue that it could previously recorded with the President.'
# output = model.generate(**tokenizer(text, return_tensors='pt'), num_beams=2, max_length=160)
# print(tokenizer.batch_decode(output, skip_special_tokens=True))

def beam(input_ids, num_beams=3):
    encoder_input_ids = input_ids
    encoder_outputs = model.get_encoder()(input_ids, output_attentions=True, output_hidden_states=True, return_dict=True)

    input_ids = model._prepare_decoder_input_ids_for_generation(input_ids, decoder_start_token_id=0)

    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(MaxLengthCriteria(max_length=40))

    beam_scorer = BeamSearchScorer(batch_size=encoder_input_ids.shape[0],
                                   num_beams=num_beams,
                                   device=model.device,
                                   length_penalty=1.0)
    
    input_ids, _ = model._expand_inputs_for_generation(input_ids, expand_size=num_beams, is_encoder_decoder=True, encoder_outputs=encoder_outputs)

    outputs = model.beam_search(input_ids=input_ids,
                           encoder_outputs=encoder_outputs,
                           beam_scorer=beam_scorer,
                           return_dict_in_generate=True,
                           stopping_criteria=stopping_criteria,
                           output_scores=True)
    return outputs

input_ids = tokenizer.batch_encode_plus(['Mohamed Hosni Mubarak, in addition to the dialogue that it could previously recorded with the President.'], padding=True, return_tensors='pt')
outputs = beam(input_ids['input_ids'], 3)
print(outputs.sequences)
# print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True))
scores = torch.stack(outputs.scores)
scores = F.softmax(scores, dim=-1)
scores = torch.transpose(scores, 0, 1)
print(torch.argmax(scores, dim=-1))