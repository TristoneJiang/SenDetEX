import torch
import torch.nn as nn
import torch.nn.functional as F
class FeatureEncoder(nn.Module):
    def __init__(self, proxy_model_embed, vocab_size):
        super().__init__()
        self.proxy_embed = proxy_model_embed  
        self.vocab_size = vocab_size
    def forward(self, sentence, regenerated, token_probs, token_logits):
        z_ins = self.proxy_embed(sentence)     
        z_inf = self.proxy_embed(regenerated)  
        probs = F.softmax(token_logits, dim=-1)  
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  
        return token_probs, entropy, z_ins, z_inf
class StyleExtractor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=d_model, kernel_size=5, groups=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        )
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8), num_layers=2
        )
        self.gate = nn.Linear(2 * d_model, d_model)
    def forward(self, token_probs, entropies):
        S = torch.stack([token_probs, entropies], dim=-1)  
        S_trans = S.transpose(0, 1).unsqueeze(0) 
        local_feat = self.local_conv(S_trans).squeeze(0).transpose(0, 1)
        global_feat = self.global_transformer(S.unsqueeze(1)).squeeze(1)
        concat = torch.cat([local_feat, global_feat], dim=-1)
        gate = torch.sigmoid(self.gate(concat))
        fused = gate * local_feat + (1 - gate) * global_feat
        z_style = fused.mean(dim=0, keepdim=True) 
        return z_style
class TripleCrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn3 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.fusion = nn.Linear(3 * d_model, d_model)
        self.classifier = nn.Linear(d_model, 1)
    def forward(self, z_style, z_ins, z_inf):
        a1, _ = self.attn1(z_style, z_ins, z_inf)
        a2, _ = self.attn2(z_ins, z_inf, z_style)
        a3, _ = self.attn3(z_inf, z_style, z_ins)
        cross = torch.cat([a1, a2, a3], dim=-1) 
        z_cross = self.fusion(cross) 
        p = torch.sigmoid(self.classifier(z_cross))  
        return p
class SenDetEX(nn.Module):
    def __init__(self, proxy_model_embed, vocab_size, d_model=768):
        super().__init__()
        self.encoder = FeatureEncoder(proxy_model_embed, vocab_size)
        self.style = StyleExtractor(d_model)
        self.fusion = TripleCrossAttention(d_model)
        self.loss_fn = nn.MSELoss()
    def forward(self, s_i, r_i, token_probs, token_logits, label):
        p_i, e_i, z_ins, z_inf = self.encoder(s_i, r_i, token_probs, token_logits)
        z_style = self.style(p_i, e_i)
        pred = self.fusion(z_style, z_ins, z_inf)
        loss = self.loss_fn(pred.squeeze(), label)
        return pred, loss

