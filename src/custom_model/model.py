import torch.nn as nn
from encoder_v3 import FeatureEncoder
from decoder import RecoloringDecoder


class RecolorizerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.decoder = RecoloringDecoder(variable_pal_size=args.variable_palette)

    def forward(self, ori_img, tgt_palette, illu):
        c1, c2, c3, c4 = self.encoder(ori_img)
        out = self.decoder(c1, c2, c3, c4, tgt_palette, illu)
        return out
    

def get_model(args):
    model = RecolorizerModel(args)
    return model