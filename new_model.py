# inspect_confidences.py
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# copy the DatasetSimple from your script or import if modularized
import re, html
from bs4 import BeautifulSoup
def clean_text(t): return re.sub(r"\s+"," ", BeautifulSoup(str(t),"html.parser").get_text()).strip()

class DatasetSimple:
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = [clean_text(t) for t in df['Review'].astype(str).tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        t = self.texts[idx]
        tok = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {'input_ids': tok['input_ids'].squeeze(0), 'attention_mask': tok['attention_mask'].squeeze(0)}
def collate_fn(batch):
    import torch
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# load tokenizer + model (use same model as in pseudo pipeline)
MODEL_NAME = "vinai/phobert-base"
# Load your most recently trained model if saved, otherwise load base encoder as probe
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
# If you saved best model state (e.g. "best_multitask_weighted.pt"), load it to a model class.
# For quick probe we load pretrained encoder + a small random head mimic so you still get embeddings.
# But better: use the model checkpoint you trained (replace below to import your model class and load_state_dict).

# --- Quick embedding-based confidence probe (faster and independent) ---
# We'll embed unlabeled texts using phoBERT and train a tiny logistic on train -> but simpler:
# Here we will just compute pseudo "confidence proxies": max softmax of encoder representation projected randomly.
# Better approach: load the actual model used for pseudo labeling and run inference.

# If you have trained model file, please update MODEL_PATH to point to it.
MODEL_PATH = "best_model_for_pseudo.pt"  # <-- replace if you have
have_model = Path(MODEL_PATH).exists()

if have_model:
    # Try to import your model class definitions (SimpleMultiTask) from script if available
    from importlib import import_module
    try:
        mmod = import_module("pseudo_labeling_safe")  # or new_model
        ModelClass = getattr(mmod, "SimpleMultiTask")
        model = ModelClass(MODEL_NAME, num_aspects=6, num_seg_classes=5)
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
    except Exception as e:
        print("Could not import model class from pseudo script; falling back to embedding probe. Error:", e)
        have_model = False

if not have_model:
    # fallback: use encoder to make embeddings and a simple classifier trained on train set would be ideal.
    # Here we'll just compute token-level representation norms as proxy of info (not ideal).
    enc = AutoModel.from_pretrained(MODEL_NAME)
    enc.eval()

# load unlabeled
unlabeled = pd.read_csv("data/problem_test.csv")
ds = DatasetSimple(unlabeled, TOKENIZER)
loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

pres_probs_all = []
seg_conf_all = []      # max softmax per-aspect (if model available)
texts = unlabeled['Review'].astype(str).tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
if have_model:
    model.to(device)
    for batch in tqdm(loader, desc="Infer unlabeled"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()           # [B,A]
            seg_probs = torch.softmax(seg_logits, dim=-1).cpu().numpy()     # [B,A,C]
            pres_probs_all.append(pres_probs)
            seg_conf_all.append(np.max(seg_probs, axis=-1))
    pres_probs_all = np.vstack(pres_probs_all)
    seg_conf_all = np.vstack(seg_conf_all)
else:
    # fallback: use encoder pooled vector norms as proxy
    enc.to(device)
    proxy = []
    for batch in tqdm(loader, desc="Embed unlabeled"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            out = enc(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:,0,:]
            proxy.append(pooled.cpu().numpy())
    emb = np.vstack(proxy)
    # make fake pres_probs (sigmoid of projection)
    rng = np.random.RandomState(0)
    W = rng.randn(emb.shape[1], 6)
    pres_probs_all = 1/(1+np.exp(-emb.dot(W)))  # not meaningful but for distribution check
    seg_conf_all = np.ones((len(emb),6)) * 0.6    # dummy constant
# Summary stats
import numpy as np
print("pres_probs stats (per-aspect):")
print("  mean:", pres_probs_all.mean(axis=0))
print("  50pct:", np.percentile(pres_probs_all,50,axis=0))
print("  75pct:", np.percentile(pres_probs_all,75,axis=0))
print("  90pct:", np.percentile(pres_probs_all,90,axis=0))
print("seg_conf (max softmax) stats (per-aspect):")
print("  mean:", seg_conf_all.mean(axis=0))
print("  50pct:", np.percentile(seg_conf_all,50,axis=0))
print("  75pct:", np.percentile(seg_conf_all,75,axis=0))
print("  90pct:", np.percentile(seg_conf_all,90,axis=0))

# show top-k candidate texts by max(pres_prob, seg_conf) for inspection
scores = np.maximum(pres_probs_all.max(axis=1), seg_conf_all.max(axis=1))
topk = 30
idxs = np.argsort(-scores)[:topk]
for rank,i in enumerate(idxs[:20],1):
    print(f"\n=== Rank {rank} score {scores[i]:.4f} ===")
    print(texts[i][:400])
    print("pres_probs:", pres_probs_all[i].round(3))
    print("seg_conf:", seg_conf_all[i].round(3))
