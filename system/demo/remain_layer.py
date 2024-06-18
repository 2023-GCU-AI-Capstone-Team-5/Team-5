import torchvision.models as models
import torch
import torch.nn as nn

def Remain_layer(data):
    Server2 = models.resnext101_32x8d(pretrained=False)

    num_features = Server2.fc.in_features
    Server2.fc = nn.Linear(num_features, 17)

    Server2.load_state_dict(torch.load("../../best_model_final.pth"))

    Server2 = torch.nn.Sequential(*list(Server2.children())[4:])
    Server2.insert(5, nn.Flatten())

    last_result = Server2(data)
    # print(last_result)
    
    pred = last_result.max(1, keepdim=True)
    # print(pred)
    return pred