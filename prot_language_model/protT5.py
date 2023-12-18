import torch
import re
print(torch.__version__)

#@title Import dependencies and check whether GPU is available. { display-mode: "form" }
from transformers import T5EncoderModel, T5Tokenizer

#import h5py
#import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))
print(T5EncoderModel)


# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.full() if device=='cpu' else model.half()

# prepare your protein sequences as a list
sequence_examples = ["PRTEINO",
                     "SEQWENCEEEEEEEEEE"]

# replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

print(sequence_examples)
import sys
sys.exit()

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
#ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")


input_ids = torch.tensor(ids['input_ids']).to(device)

attention_mask = torch.tensor(ids['attention_mask']).to(device)



# generate embeddings
with torch.no_grad():
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

# extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
emb_0 = embedding_repr.last_hidden_state[0,:7] # shape (7 x 1024)
# same for the second ([1,:]) sequence but taking into account different sequence lengths ([1,:8])
emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)

# if you want to derive a single representation (per-protein embedding) for the whole protein
emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)

print(emb_0)
print(emb_0.size())

print(embedding_repr)

print(attention_mask)
print(attention_mask.size())