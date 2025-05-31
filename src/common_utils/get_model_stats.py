import sys
sys.path.insert(0, "../custom_model/")
from model import get_model

def get_number_trainable_params():
    model = get_model()
    trainable_params =sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = trainable_params / 1_000_000
    return trainable_params

if __name__=="__main__":
    params = get_number_trainable_params()
    print("Number of trainable parameters:", params, "Million")