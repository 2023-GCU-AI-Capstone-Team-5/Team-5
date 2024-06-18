import os
import time

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import ConcatDataset
from sklearn.metrics import f1_score

# fix seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Write to wandb
import wandb
wandb.init(project='Face Classification')

#Name to be recorded
wandb.run.name = 'resnext101_32x8d'
wandb.run.save()

# Hyperparameters
BATCH_SIZE = 12
EPOCH = 30
lr = 0.001

OPTIMIZER = 'SGD'

 
args = {
    "learning_rate": lr,
    "epochs": EPOCH,
    "batch_size": BATCH_SIZE,
    "optimizer": OPTIMIZER,
}
wandb.config.update(args)


### GPU Setting ###
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(DEVICE)

### Custom Dataset ###
class CFDataset(Dataset):
    def __init__(self, transform, mode='train'):
        self.transform = transform
        self.mode = mode
        
        self.celeb_folder = sorted(os.listdir("data/Celebrity Faces Dataset"))
        
        self.image_folder = []
        self.label = []
        for idx, celeb in enumerate(self.celeb_folder):
            celeb_image = os.listdir("data/Celebrity Faces Dataset"+"/"+celeb)
            for img in celeb_image:
                prefix = int(img.split("_")[0])
                if len(celeb_image) != 200:
                    if self.mode == 'train' and (prefix <= 80):
                        self.image_folder.append(img)
                        self.label.append(idx)
                    elif self.mode == 'valid' and (80 < prefix and prefix <=90):
                        self.image_folder.append(img)
                        self.label.append(idx)
                    elif self.mode == 'test' and (90 < prefix):
                        self.image_folder.append(img)
                        self.label.append(idx)
                else:
                    if self.mode == 'train' and (prefix <= 160):
                        self.image_folder.append(img)
                        self.label.append(idx)
                    elif self.mode == 'valid' and (160 < prefix and prefix <=180):
                        self.image_folder.append(img)
                        self.label.append(idx)
                    elif self.mode == 'test' and (180 < prefix):
                        self.image_folder.append(img)
                        self.label.append(idx)
    
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        img_path = self.image_folder[idx]
        img = Image.open(os.path.join("data/Celebrity Faces Dataset", self.celeb_folder[self.label[idx]], img_path)).convert("RGB")
        img = self.transform(img)
        
        return (img, self.label[idx])
    
### Data Preprocessing ###
transforms_train = transforms.Compose([transforms.Resize((592, 474)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]), ])
transforms_valtest = transforms.Compose([transforms.Resize((592, 474)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),])


train_set = CFDataset(mode='train', transform=transforms_train)
val_set = CFDataset(mode='valid', transform=transforms_valtest)
test_set = CFDataset(mode='test', transform=transforms_valtest)

print('Num of each dataset:', len(train_set), len(val_set), len(test_set))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print('Loaded dataloader')



### Model / Optimizier ###
# model = models.resnet18(pretrained=True)
model = models.resnext101_32x8d(pretrained=True)
# model = models.wide_resnet101_2(pretrained=True)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 17)
# half_in_size = round(num_features/2)
# layer_width = 512
# Num_class=17
# import spinal.spinalnet as spinalnet
# model.fc = spinalnet.SpinalNet(half_in_size=half_in_size, layer_width=layer_width, Num_class=Num_class)

model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print('Created a learning model and optimizer')

### Train/Evaluation ###
def train(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        optimizer.zero_grad()
        train_loss = F.cross_entropy(output, target).to(DEVICE)
        
        train_loss.backward()
        optimizer.step()
        
        running_loss += train_loss.item()
        if i == 0:
            print(f'Train Epoch : {epoch} [{i}/{len(train_loader)}]\tLosss: {train_loss.item():.6f}')
            wandb.log({"Training loss": running_loss / 10, 'epoch': epoch})
            running_loss = 0.0
            
    return train_loss

def evaluate(model, val_loader, epoch, mode='valid'):
    model.eval()
    eval_loss = 0
    correct = 0
    if mode == "test":
        confusion_target = []
        confusion_pred = []
        confusion_probas = []
    with torch.no_grad():
        for i, (image, target) in enumerate(val_loader):
            image, target = image.to(DEVICE), target.to(DEVICE)
            output = model(image)
            
            eval_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if mode == "test":
                confusion_target.append(target.cpu().numpy())
                confusion_pred.append(pred.cpu().numpy().flatten())
                confusion_probas.append(output.cpu().numpy())
    if mode == "test":
        target_log = np.concatenate(confusion_target)
        pred_log = np.concatenate(confusion_pred)
        probas_log = np.concatenate(confusion_probas)
        classes_name = list(range(17))
        wandb.log({"Test F1 score": f1_score(target_log, pred_log, average="macro")})
        wandb.log({"roc" : wandb.plot.roc_curve(target_log, probas_log, classes_name)})
        wandb.log({'pr': wandb.plot.pr_curve(target_log, probas_log, classes_name)})
        
    eval_loss /= len(val_loader.dataset)
    eval_accuracy = 100 * correct /len(val_loader.dataset)
    return eval_loss, eval_accuracy

### Main ###
start = time.time()
best = 0
best_epoch = 0
for epoch in range(EPOCH):
    train_loss = train(model, train_loader, optimizer, epoch)
    val_loss, val_accuracy = evaluate(model, val_loader, epoch)
    
    # Save best model
    if val_accuracy > best:
        best = val_accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), "./best_model_final.pth")
    wandb.log({"Validation Loss": val_loss, 'epoch': epoch})
    wandb.log({"Validation Accuracy": val_accuracy, 'epoch': epoch})
    print(f'[{epoch}] Validation Loss : {val_loss:.4f}, Accuracy: {val_accuracy:.4f}%')

wandb.log({"Validation Best Epoch": best_epoch})

# Load test model
# net = models.resnet18(pretrained=False)
net = models.resnext101_32x8d(pretrained=False)
# net = models.wide_resnet101_2(pretrained=False)

net_num_features = net.fc.in_features
net.fc = nn.Linear(net_num_features, 17)
# net.fc = spinalnet.SpinalNet(half_in_size=half_in_size, layer_width=layer_width, Num_class=Num_class)

net.load_state_dict(torch.load("./best_model_final.pth"))
net.to(DEVICE)

# Test result
test_loss, test_accuracy = evaluate(net, test_loader, 0, mode='test')
wandb.log({"Test loss": test_loss})
wandb.log({"Test Accuracy": test_accuracy})
print(f'[FINAL] Test Loss : {test_loss:.4f}, Accuracy: {test_accuracy:.4f}%')

end = time.time()
elasped_time = end - start
wandb.log({"Elasped time": elasped_time})
print("Best Accuracy: ", best)
print(f"Elasped Time: {int(elasped_time/3600)}h, {int(elasped_time/60)}m, {int(elasped_time%60)}s")