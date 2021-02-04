import math
import torch
import os
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling_openai import OpenAIGPTConfig, WEIGHTS_NAME, CONFIG_NAME

model_path = 'openai-gpt'
output_dir = './language-quality-subreward/gpt_output'
WEIGHTS_NAME = 'pytorch_model.bin'
special_tokens = ['_start_', '_delimiter_', '_classify_']
# Load pre-trained model (weights)
with torch.no_grad():
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    config = OpenAIGPTConfig(output_config_file)

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    model_state_dict = torch.load(output_model_file, map_location='cpu')
    model = OpenAIGPTLMHeadModel(config)
    model.load_state_dict(model_state_dict)

    # model = OpenAIGPTLMHeadModel.from_pretrained(model_path)
    # model.load_state_dict(torch.load(output_model_file, map_location='cpu'))
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_path, cache_dir='./tmp/', special_tokens=special_tokens)

'''
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
'''


def get_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss.item())

'''
a=['there is a book on the desk',
                'there is a plane on the desk',
                        'there is a book in the desk']
print([get_score(i) for i in a])
'''

sentence1 = 'hi how are you doing?'
sentence2 = 'hi how are you doing doing doing doing doing doing doing?'
sentence3 = 'hunting hunting hunting hunting hunting hunting hunting hunting hunting hunting hunting'
sentence4 = 'i am alright i fix airplanes and listen to vinyl music and drive junk cars and drive junk cars.'
sentence5 = 'He is go to school.'
sentence6 = 'He is going to school.'
sentence7 = 'i am doing well how are you? please tell me more about yourself.'
sentence8 = 'i listen to everything ; although the vinyl records, which i really like, are my favorite ; any'
sentence9 = 'i fix airplanes sometimes lol i also fix them and fix them haha lol lol lol'
sentence10 = 'hey!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
score1 = get_score(sentence1)
score2 = get_score(sentence2)
score3 = get_score(sentence3)
score4 = get_score(sentence4)
score5 = get_score(sentence5)
score6 = get_score(sentence6)
score7 = get_score(sentence7)
score8 = get_score(sentence8)
score9 = get_score(sentence9)
score10 = get_score(sentence10)
print('score1 =', score1)
print('score2 =', score2)
print('score3 =', score3)
print('score4 =', score4)
print('score5 =', score5)
print('score6 =', score6)
print('score7 =', score7)
print('score8 =', score8)
print('score9 =', score9)
print('score10 =', score10)

''' Expected output
score1 = 6.07521112989909
score2 = 486.5057432595975
score3 = 256620.28555512003
score4 = 66.2635356204048
score5 = 116.4927119247885
score6 = 24.789758944543998
score7 = 7.277927336245507
score8 = 74.70515071338066
score9 = 129.93358925186743
score10 = 1.5633166462070827
'''