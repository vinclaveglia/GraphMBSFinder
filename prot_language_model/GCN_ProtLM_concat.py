import torch
from base.wp_model import ProteinSegmenter2
from base.input import get_random_train_test_proteins
from base.PDB_MBS_preprocessing import get_dataset
import re
import numpy as np
from base import config
import pathlib

"""
Produce gli embedding con il prot language model e li salva su disco
"""

from base.pdb_loader import MAPPING
aa_map = {}
for name, letter in MAPPING:
    aa_map[name] = letter

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

#net2 = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = 50)
#net2.load_state_dict(torch.load('metadata/model_0.01_50_2_5_0.pth', map_location=device))
#print(net2)

# todo -- prendere quelle usate per la GCN, non cagare il cazzo... --

train_proteins, test_proteins, all_sites = get_random_train_test_proteins()

#caricare le sequenze da dare in input al transformer
print(test_proteins[:10])
#print(all_sites)

dfs = get_dataset(test_proteins + train_proteins, all_sites, dataframe_only=True)

sequences = []
for prot_name, prot_df in dfs:
    print(f"------ {prot_name} ---------")
    sequence = prot_df.residue_name.tolist() # lista

    if len(sequence) > 1000:
        print(">> too long ", len(sequence), "SKIP")
        continue

    sequence = ''.join([aa_map[x] for x in sequence]) # ora è stringa
    sequence = re.sub(r"[UZOB]", "X", sequence) # sempre stringa
    sequence = ' '.join(list(sequence))

    sequences.append(
        (prot_name, sequence)
    )


from transformers import T5EncoderModel, T5Tokenizer

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

#sequences = [x[1] for x in sequences]
embeddings_path = pathlib.Path(config.ProtT5_EMBEDDINGS_PATH)
embeddings_path.mkdir(exist_ok=True)


for j_idx, (prot_name, sequence) in enumerate(sequences):
    print(f"{j_idx+1}/{len(sequences)} (len {len(sequence.split())})", prot_name)
    ids = tokenizer([sequence], add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)

    attention_mask = torch.tensor(ids['attention_mask']).to(device) # serve per gestire il padding

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)


    emb_0 = embedding_repr.last_hidden_state

    np.save(f'{embeddings_path}/{prot_name[:4]}.protT5', emb_0.cpu().numpy())

    #prova = np.load(f'{embeddings_path}/{prot_name[:4]}.protT5.npy')

    #print(prova.shape)