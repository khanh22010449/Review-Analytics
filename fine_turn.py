# finetune_peft_lora_0to5.py
"""
Fine-tune multi-task (presence + per-aspect sentiment) using LoRA/PEFT.
Sentiment classes: 0..5 (6 classes). Presence target computed as (rating > 0).
- Apply LoRA to encoder (AutoModel) then train heads + LoRA adapters.
- Safe encoder forward and attention pool (fp16-safe).
"""

import os, math, random
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from torch.optim import AdamW

# PEFT
try:
    from peft import LoraConfig, get_peft_model
except Exception as e:
    raise ImportError("Please install peft (pip install peft). Error: " + str(e))

# ---------------- CONFIG ----------------
MODEL_NAME = "xlm-roberta-base"   # change if you prefer
CSV_PATH = "./train-problem.csv"  # your training csv (must contain 'Review' + ASPECT_COLS)
ASPECT_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 60
HEAD_LR = 2e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IGNORE_INDEX = None   # not used; we treat 0 as a valid class
ACCUMULATION_STEPS = 1
USE_AMP = True
FOCAL_GAMMA = 2.0
PATIENCE = 60
SAVE_PATH = "../lora_multitask_best.pth"

# LoRA config
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "key", "value", "dense", "q_proj", "k_proj", "v_proj", "o_proj", "o_dense"]

# stability clips
POS_WEIGHT_CLIP = (1e-2, 100.0)
CLASS_WEIGHT_CLIP = (0.2, 10.0)
NUM_RATING_CLASSES = 6  # 0..5
# ----------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# ---------- Dataset ----------
class ReviewDataset(Dataset):
    def __init__(self, texts, rating_arr, tokenizer):
        # rating_arr: (N, K) values in 0..5
        self.texts = texts
        self.rating = rating_arr.astype(np.int64)
        self.presence = (self.rating > 0).astype(np.float32)  # presence = rating > 0
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = str(self.texts[idx])
        enc = self.tokenizer(t, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['presence'] = torch.tensor(self.presence[idx], dtype=torch.float)
        item['rating'] = torch.tensor(self.rating[idx], dtype=torch.long)  # 0..5
        return item

# ---------- AttentionPool ----------
class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, hidden_states, mask):
        scores = self.linear(hidden_states).squeeze(-1)
        mask_bool = (mask.to(dtype=torch.bool, device=scores.device))
        if scores.dtype in (torch.float16, torch.bfloat16):
            neg_inf_val = -1e4
        else:
            neg_inf_val = -1e9
        neg_inf = torch.tensor(neg_inf_val, dtype=scores.dtype, device=scores.device)
        scores = scores.masked_fill(~mask_bool, neg_inf)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * hidden_states).sum(dim=1)
        return pooled

# ---------- safe encoder forward ----------
def safe_encoder_forward(encoder_module, **kwargs):
    try:
        return encoder_module(**kwargs)
    except TypeError as e:
        msg = str(e).lower()
        if "unexpected keyword argument 'labels'" in msg or 'labels' in kwargs:
            kwargs_filtered = {k:v for k,v in kwargs.items() if k != 'labels'}
            if hasattr(encoder_module, "base_model"):
                try:
                    return encoder_module.base_model(**kwargs_filtered)
                except Exception:
                    pass
            return encoder_module(**kwargs_filtered)
        else:
            raise

# ---------- Multi-task model ----------
class MultiTaskModelForPEFT(nn.Module):
    def __init__(self, encoder, hidden_size, num_aspects, num_rating_classes=NUM_RATING_CLASSES, hidden_dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.att_pool = AttentionPool(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)
        self.shared_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(hidden_dropout))
        self.presence_head = nn.Linear(hidden_size, num_aspects)
        self.sentiment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size//2, num_rating_classes)
            ) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = safe_encoder_forward(self.encoder, input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last = out.last_hidden_state
        pooled = self.att_pool(last, attention_mask)
        pooled = self.dropout(pooled)
        shared = self.shared_proj(pooled)
        pres_logits = self.presence_head(shared)
        sent_logits = torch.stack([h(shared) for h in self.sentiment_heads], dim=1)  # (B, K, C)
        return pres_logits, sent_logits

# ---------- Loss/Utils ----------
class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.base = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:
            self.base = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        bce_loss = self.base(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        mod = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = mod * alpha_t * bce_loss
        else:
            loss = mod * bce_loss
        return loss.mean()

def compute_presence_pos_weight(pres_arr):
    pos = pres_arr.sum(axis=0)
    neg = pres_arr.shape[0] - pos
    pos = np.where(pos == 0, 1.0, pos)
    pos_weight = (neg / pos).astype(np.float32)
    pos_weight = np.clip(pos_weight, POS_WEIGHT_CLIP[0], POS_WEIGHT_CLIP[1])
    return torch.tensor(pos_weight, dtype=torch.float32)

def compute_sentiment_class_weights(rating_arr, num_classes=NUM_RATING_CLASSES):
    flat = rating_arr.ravel()
    counts = Counter(flat.tolist())
    total = sum(counts.values()) if counts else 1
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt == 0:
            weights.append(1.0)
        else:
            weights.append(float(total) / (num_classes * cnt))
    weights = np.array(weights, dtype=np.float32)
    weights = np.clip(weights, CLASS_WEIGHT_CLIP[0], CLASS_WEIGHT_CLIP[1])
    return torch.tensor(weights, dtype=torch.float32)

# ---------- Eval ----------
def evaluate(model, loader, threshold=0.5):
    model.eval()
    preds_presence, trues_presence = [], []
    preds_rating, trues_rating = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            presence = batch['presence'].to(DEVICE)
            rating = batch['rating'].to(DEVICE)
            pres_logits, sent_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pres_prob = torch.sigmoid(pres_logits).cpu().numpy()
            pres_pred = (pres_prob >= threshold).astype(int)
            preds_presence.append(pres_pred); trues_presence.append(presence.cpu().numpy())
            sent_pred = torch.argmax(sent_logits, dim=-1).cpu().numpy()
            preds_rating.append(sent_pred); trues_rating.append(rating.cpu().numpy())
    import numpy as np
    preds_presence = np.vstack(preds_presence); trues_presence = np.vstack(trues_presence)
    preds_rating = np.vstack(preds_rating); trues_rating = np.vstack(trues_rating)
    f1_macro = f1_score(trues_presence, preds_presence, average='macro', zero_division=0)
    f1_micro = f1_score(trues_presence, preds_presence, average='micro', zero_division=0)
    accs=[]; maes=[]
    for k in range(trues_rating.shape[1]):
        accs.append(accuracy_score(trues_rating[:,k], preds_rating[:,k]))
        maes.append(mean_absolute_error(trues_rating[:,k], preds_rating[:,k]))
    avg_acc = np.mean(accs) if accs else float('nan')
    avg_mae = np.mean(maes) if maes else float('nan')
    return {"f1_macro_presence": f1_macro, "f1_micro_presence": f1_micro, "avg_rating_acc": avg_acc, "avg_rating_mae": avg_mae}

# ---------- Train loop ----------
def train_loop(model, train_loader, val_loader, optimizer, scheduler, pos_weight, class_weight,
               num_epochs=EPOCHS, accumulation_steps=ACCUMULATION_STEPS, use_amp=USE_AMP):
    best_metric = -1.0
    no_improve = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    focal_bce = FocalBCEWithLogits(gamma=FOCAL_GAMMA, pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None)
    ce_loss = nn.CrossEntropyLoss(weight=class_weight.to(DEVICE) if class_weight is not None else None)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE); attention_mask = batch['attention_mask'].to(DEVICE)
            presence = batch['presence'].to(DEVICE); rating = batch['rating'].to(DEVICE)
            if scaler:
                with torch.cuda.amp.autocast():
                    pres_logits, sent_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss_pres = focal_bce(pres_logits, presence)
                    loss_sent = 0.0
                    K = sent_logits.size(1)
                    for k in range(K):
                        loss_sent = loss_sent + ce_loss(sent_logits[:,k,:], rating[:,k])
                    loss_sent = loss_sent / K
                    loss = loss_pres + loss_sent
                scaler.scale(loss/accumulation_steps).backward()
                if (step+1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                    if scheduler: scheduler.step()
                    optimizer.zero_grad()
            else:
                pres_logits, sent_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_pres = focal_bce(pres_logits, presence)
                loss_sent = 0.0
                K = sent_logits.size(1)
                for k in range(K):
                    loss_sent = loss_sent + ce_loss(sent_logits[:,k,:], rating[:,k])
                loss_sent = loss_sent / K
                loss = loss_pres + loss_sent
                (loss/accumulation_steps).backward()
                if (step+1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler: scheduler.step()
                    optimizer.zero_grad()
            total_loss += loss.item() * input_ids.size(0)
        avg_train_loss = total_loss / (len(train_loader.dataset))
        val_metrics = evaluate(model, val_loader)
        print(f"[Epoch {epoch+1}] train_loss={avg_train_loss:.4f} val_metrics={val_metrics}")
        score = val_metrics.get('avg_rating_acc', float('nan'))
        if math.isnan(score):
            score = val_metrics['f1_micro_presence']
        if score > best_metric:
            best_metric = score
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best -> {SAVE_PATH} (score={score:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break
    return best_metric

# ---------- Data prepare ----------
def prepare_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    if 'Review' not in df.columns:
        raise ValueError("CSV must contain column 'review'")
    texts = df['Review'].astype(str).tolist()
    sample = df[ASPECT_COLS].copy()
    # Expect sample values are integers in 0..5
    rating_arr = sample.values.astype(int)
    # sanity check
    if np.any(rating_arr < 0) or np.any(rating_arr > (NUM_RATING_CLASSES - 1)):
        raise ValueError("Found rating values outside 0..{}".format(NUM_RATING_CLASSES - 1))
    return texts, rating_arr

# ---------- Main ----------
def main():
    texts, rating_arr = prepare_data(CSV_PATH)
    X_train, X_val, r_train, r_val = train_test_split(texts, rating_arr, test_size=0.1, random_state=SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(X_train, r_train, tokenizer)
    val_ds = ReviewDataset(X_val, r_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print("Loading base encoder:", MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    hidden_size = encoder.config.hidden_size

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none"
    )
    print("Applying LoRA to encoder...")
    encoder = get_peft_model(encoder, lora_config)

    model = MultiTaskModelForPEFT(encoder=encoder, hidden_size=hidden_size, num_aspects=len(ASPECT_COLS), num_rating_classes=NUM_RATING_CLASSES)
    model.to(DEVICE)

    # show trainable params
    try:
        if hasattr(encoder, "print_trainable_parameters"):
            encoder.print_trainable_parameters()
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable {trainable}/{total} params")
    except Exception:
        pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    total_steps = math.ceil(len(train_loader)/ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    # compute pos/class weight
    p_train = (r_train > 0).astype(np.float32)
    pos_weight = compute_presence_pos_weight(p_train)
    class_weight = compute_sentiment_class_weights(r_train, num_classes=NUM_RATING_CLASSES)
    print("pos_weight:", pos_weight.numpy(), "class_weight:", class_weight.numpy())

    best = train_loop(model, train_loader, val_loader, optimizer, scheduler, pos_weight, class_weight,
                      num_epochs=EPOCHS, accumulation_steps=ACCUMULATION_STEPS, use_amp=USE_AMP)
    print("Best val metric:", best)

    # save adapter and full state
    try:
        encoder.save_pretrained("peft_lora_multitask_encoder_0to5")
        print("Peft adapters saved to peft_lora_multitask_encoder_0to5/")
    except Exception as e:
        print("Could not save peft adapters:", e)

    torch.save(model.state_dict(), "lora_multitask_0to5_model_full.pth")
    print("Saved full model state to lora_multitask_0to5_model_full.pth")

if __name__ == "__main__":
    main()
