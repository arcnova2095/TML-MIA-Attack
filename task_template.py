import os
import sys
import torch
import pandas as pd
import numpy as np
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from xgboost import XGBClassifier
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "93684f2bf1d14acedf752c31dbecc931"
TASK_ID = "01-mia"  #DONOT CHANGE



# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)


# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform


# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(device)
model.eval()
def collate(batch):
    ids= [item[0] for item in batch]
    imgs= torch.stack([item[1] for item in batch])
    labels= torch.tensor([item[2] if item[2]!= None else -1 for item in batch])
    if len(batch[0])==4:
        membership= torch.tensor([item[3] if item[3] is not None else -1 for item in batch])
        return ids,imgs, labels, membership
    return ids, imgs, labels
##loading Dataset
pub_loader= DataLoader(pub_ds,batch_size=64,collate_fn=collate )
pri_loader= DataLoader(priv_ds,batch_size=64,collate_fn=collate )
# device= torch.device("cuda")

##extract pub, priv features

def feature_extract(loader,model):
    features,ids, memberships=[],[],[]
    confidences=[]
    losses=[]
    entropys=[]
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in loader:
            if len(batch)==4:
                id, img, _,membership=batch
            else:
                id,img,_= batch
                membership=torch.full((len(img),),-1)
            img= img.to(device)
            logits= model(img)
            probs= F.softmax(logits,dim=1)

            confidence,pseudo= probs.max(dim=1)
            entropy = -(probs*torch.log(probs+1e-10)).sum(dim=1)
            loss= criterion(logits,pseudo)

            ids.extend(id)
            memberships.extend(membership.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
            entropys.extend(entropy.cpu().numpy())
            losses.extend(loss.cpu().numpy())
        X= np.array([confidences,entropys,losses]).T
        Y= np.array(memberships)
        return ids,X,Y
    
_,X_train, Y_train= feature_extract(pub_loader,model=model)
print("pub extraction finished")
priv_ids,X_test, _=feature_extract(pri_loader,model=model)
print("priv extraction finished")

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, 
    Y_train, 
    test_size=0.2,       # 20% goes to validation, 80% stays for training
    random_state=42,     # Ensures you get the same split every time you run it
    stratify=Y_train     # Crucial for classification: ensures both sets have the same ratio of 0s and 1s
)
##XGBOOST
scale_pos_weight = (len(Y_train) - sum(Y_train)) / sum(Y_train)
print("XGBOOST Classifier Training")
classifier_model= XGBClassifier(
                n_estimators=10000,
                max_depth=15,
                learning_rate=0.015,
                gamma=0.2,
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=0.2,
                reg_lambda=1.5,
                scale_pos_weight=scale_pos_weight,
                tree_method='hist',
                device='cuda',
                eval_metric='auc',
                early_stopping_rounds=100,
                random_state=42,
                growth_policy= 'lossguide'
)

classifier_model.fit(X_train, 
    Y_train,
    eval_set=[(X_val, y_val)],
    verbose=50)

##predicitions 
final_scores= classifier_model.predict_proba(X_test)[:,1]

df = pd.DataFrame({"id": priv_ids, "score": final_scores})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Success! Saved simplified predictions to: {OUTPUT_CSV}")
            



# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)