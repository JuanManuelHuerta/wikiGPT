from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import psutil, os
from tqdm.auto import tqdm
import keyword


'''

download_config = DownloadConfig(delete_extracted=True)
remote_dataset = load_dataset('transformersbook/codeparrot', split="train",
                              streaming=True)

def tok_list(tokenizer, string):
    input_ids = tokenizer(string, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode(tok) for tok in input_ids]

tokenizer_T5 = AutoTokenizer.from_pretrained("t5-base")
tokenizer_camembert = AutoTokenizer.from_pretrained("camembert-base")

print(f'T5 tokens for "sex": {tok_list(tokenizer_T5,"sex")}')
print(f'CamemBERT tokens for "being": {tok_list(tokenizer_camembert,"being")}')

'''

python_code = r"""def say_hello():
    print("Hello, World!")
# Print it
say_hello()
"""



tokenizer = AutoTokenizer.from_pretrained("gpt2")

print(tokenizer(python_code).tokens())


byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v, k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())
print(f'Size of our base vocabulary: {len(base_vocab)}')
print(f'First element: `{base_vocab[0]}`, last element: `{base_vocab[-1]}`')



## This is done in the CPU

#length = 10000000
length = 1000000



#dataset=load_dataset("wikipedia", "20220301.en",split="train", streaming=True)
dataset=load_dataset("wikipedia", "20220301.en",split="train")
#dataset_name = 'transformersbook/codeparrot-train'
#dataset = load_dataset(dataset_name, split="train", streaming=True)
iter_dataset = iter(dataset)

def batch_iterator(batch_size=100):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)['text'] for _ in range(batch_size)]

new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), 
                                                  vocab_size=128000,
                                                  initial_alphabet=base_vocab)
print(new_tokenizer(python_code).tokens())


i=0
for w in new_tokenizer.vocab:
    print(w)
    if i>=10:
        break
    i+=1
    
#print(f'There are in total {len(keyword.kwlist)} Python keywords.')
#for keyw in keyword.kwlist:
#    if keyw not in new_tokenizer.vocab:
#        print(f'No, keyword `{keyw}` is not in the vocabulary')


model_ckpt = "wikipedia_english"
org = ""
new_tokenizer.push_to_hub(model_ckpt, organization=org)

