from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_causal_mask(seq_len: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
  return torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1).bool()

def _expand_attn_mask(attn_mask: torch.Tensor, batch_size: int, num_heads: int) -> torch.Tensor:
  # attn_mask can be of shape (seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k) or (batch_size, num_heads, seq_len_q, seq_len_k)
  if attn_mask.dim() == 2:
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len_q, seq_len_k)
    attn_mask = attn_mask.expand(batch_size, num_heads, -1, -1) # (batch_size, num_heads, seq_len_q, seq_len_k)
  elif attn_mask.dim() == 3:
    attn_mask = attn_mask.unsqueeze(1) # (batch_size, 1, seq_len_q, seq_len_k)
    attn_mask = attn_mask.expand(-1, num_heads, -1, -1) # (batch_size, num_heads, seq_len_q, seq_len_k)
  elif attn_mask.dim() == 4:
    pass # already in the correct shape (batch_size, num_heads, seq_len_q, seq_len_k)
  else:
    raise ValueError(f"attn_mask must be of shape (seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k), but got {attn_mask.shape}")
  return attn_mask

def build_combined_attn_mask(
    attn_mask: Optional[torch.Tensor], 
    key_padding_mask: Optional[torch.Tensor], 
    batch_size: int, 
    num_heads: int, 
    seq_len_q: int,
    seq_len_k: int,
    device: torch.device = torch.device('cpu')) -> Optional[torch.Tensor]:
    combined = None
    if attn_mask is not None:
      attn_mask = attn_mask.to(device)
      combined = _expand_attn_mask(attn_mask, batch_size, num_heads)
      if combined.size(-2) != seq_len_q or combined.size(-1) != seq_len_k:
        raise ValueError(f"attn_mask has shape {attn_mask.shape} but expected ({seq_len_q}, {seq_len_k}) or ({batch_size}, {seq_len_q}, {seq_len_k}) or ({batch_size}, {num_heads}, {seq_len_q}, {seq_len_k})")

    if key_padding_mask is not None:
      if key_padding_mask.dim() != 2 or key_padding_mask.size(0) != batch_size or key_padding_mask.size(1) != seq_len_k:
        raise ValueError(f"key_padding_mask must be of shape (batch_size, seq_len_k) but got {key_padding_mask.shape}")
      key_padding_mask = key_padding_mask.to(device)
      key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, seq_len_k)
      key_padding_mask = key_padding_mask.expand(-1, num_heads, seq_len_q, -1) # (batch_size, num_heads, seq_len_q, seq_len_k)
      combined = key_padding_mask if combined is None else (combined | key_padding_mask)

    return combined

class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size: int, d_model: int):
    super(TokenEmbedding, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.embedding(x)

class SinusoidalPositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_len: int = 5000):
    super(SinusoidalPositionalEncoding, self).__init__()
    self.max_len = max_len
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # (d_model/2,)
    pe[:, 0::2] = torch.sin(position * div_term) # (max_len, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term) # (max_len, d_model/2)
    pe = pe.unsqueeze(0) # (1, max_len, d_model)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    _, seq_len, _ = x.size()
    if seq_len > self.max_len:
      raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
    return self.pe[:, :seq_len, :] + x

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
    super(MultiHeadAttention, self).__init__()
    if d_model % num_heads != 0:
      raise ValueError(f"d_model must be divisible by num_heads, but got d_model={d_model} and num_heads={num_heads}")
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)

    self.attn_drop = nn.Dropout(dropout)
    self.proj_drop = nn.Dropout(dropout)

  def forward(
      self, 
      query: torch.Tensor, 
      key: torch.Tensor, 
      value: torch.Tensor, 
      attn_mask: Optional[torch.Tensor] = None, 
      key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    batch_size = query.size(0)

    # Linear projections
    Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, seq_len_q, d_k)
    K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, seq_len_k, d_k)
    V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, seq_len_v, d_k)

    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k).float()) # (batch_size, num_heads, seq_len_q, seq_len_k)

    combined_mask = build_combined_attn_mask(
      attn_mask=attn_mask, 
      key_padding_mask=key_padding_mask, 
      batch_size=batch_size, 
      num_heads=self.num_heads, 
      seq_len_q=query.size(1), 
      seq_len_k=key.size(1),
      device=scores.device
    )

    if combined_mask is not None:
      scores = scores.masked_fill(combined_mask, float('-inf'))

    attn = F.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len_q, seq_len_k)
    attn = self.attn_drop(attn)

    output = torch.matmul(attn, V) # (batch_size, num_heads, seq_len_q, d_k)
    output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # (batch_size, seq_len_q, d_model)
    output = self.out_proj(output)
    output = self.proj_drop(output)

    return output

class FeedForward(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
    super(FeedForward, self).__init__()
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc1(x)
    x = F.gelu(x)
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x

class EncoderLayer(nn.Module):
  def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, pre_ln: bool = False):
    super(EncoderLayer, self).__init__()
    self.pre_ln = pre_ln
    self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
    self.ffn = FeedForward(d_model, d_ff, dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if self.pre_ln:
      # Pre-LN: LayerNorm before attention and FFN
      x = x + self.drop1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)) # Residual connection
      x = x + self.drop2(self.ffn(self.norm2(x))) # Residual connection
      return x
    else:
      # Post-LN: LayerNorm after attention and FFN
      x = self.norm1(x + self.drop1(self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask))) # Residual connection
      x = self.norm2(x + self.drop2(self.ffn(x))) # Residual connection
      return x

class DecoderLayer(nn.Module):
  def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, pre_ln: bool = False):
    super(DecoderLayer, self).__init__()
    self.pre_ln = pre_ln
    self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
    self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
    self.ffn = FeedForward(d_model, d_ff, dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)
    self.drop3 = nn.Dropout(dropout)

  def forward(
      self, 
      x: torch.Tensor, 
      enc_output: torch.Tensor, 
      tgt_attn_mask: Optional[torch.Tensor] = None, 
      tgt_key_padding_mask: Optional[torch.Tensor] = None,
      memory_attn_mask: Optional[torch.Tensor] = None,
      memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if self.pre_ln:
      # Pre-LN: LayerNorm before attention and FFN
      x = x + self.drop1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=tgt_attn_mask, key_padding_mask=tgt_key_padding_mask)) # Residual connection
      x = x + self.drop2(self.cross_attn(self.norm2(x), enc_output, enc_output, attn_mask=memory_attn_mask, key_padding_mask=memory_key_padding_mask)) # Residual connection
      x = x + self.drop3(self.ffn(self.norm3(x))) # Residual connection
      return x
    else:
      # Post-LN: LayerNorm after attention and FFN
      x = self.norm1(x + self.drop1(self.self_attn(x, x, x, attn_mask=tgt_attn_mask, key_padding_mask=tgt_key_padding_mask))) # Residual connection
      x = self.norm2(x + self.drop2(self.cross_attn(x, enc_output, enc_output, attn_mask=memory_attn_mask, key_padding_mask=memory_key_padding_mask))) # Residual connection
      x = self.norm3(x + self.drop3(self.ffn(x))) # Residual connection
      return x

@dataclass
class TransformerConfig:
  d_model: int = 512
  num_heads: int = 8
  d_ff: int = 2048
  num_layers: int = 6
  dropout: float = 0.1

  pre_ln: bool = False
  pos_type: str = 'sinusoidal'
  max_len: int = 512

class EncoderBackbone(nn.Module):
  def __init__(self, cfg: TransformerConfig):
    super(EncoderBackbone, self).__init__()
    self.cfg = cfg
    self.pos = self._build_pos(cfg.pos_type, cfg.max_len, cfg.d_model)
    self.drop = nn.Dropout(cfg.dropout)
    self.layers = nn.ModuleList([EncoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout, cfg.pre_ln) for _ in range(cfg.num_layers)])
    self.final_ln = nn.LayerNorm(cfg.d_model) if cfg.pre_ln else nn.Identity()
  
  def _build_pos(self, pos_type: str, max_len: int, d_model: int) -> nn.Module:
    if pos_type == 'sinusoidal':
      return SinusoidalPositionalEncoding(d_model, max_len)
    else:
      raise ValueError(f"Unsupported pos_type: {pos_type}")
  
  def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if self.pos is not None:
      x = self.pos(x)
    x = self.drop(x)
    for layer in self.layers:
      x = layer(x, key_padding_mask=key_padding_mask)
    x = self.final_ln(x)
    return x

class DecoderBackbone(nn.Module):
  def __init__(self, cfg: TransformerConfig):
    super(DecoderBackbone, self).__init__()
    self.cfg = cfg
    self.pos = self._build_pos(cfg.pos_type, cfg.max_len, cfg.d_model)
    self.drop = nn.Dropout(cfg.dropout)
    self.layers = nn.ModuleList([DecoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout, cfg.pre_ln) for _ in range(cfg.num_layers)])
    self.final_ln = nn.LayerNorm(cfg.d_model) if cfg.pre_ln else nn.Identity()
  
  def _build_pos(self, pos_type: str, max_len: int, d_model: int) -> nn.Module:
    if pos_type == 'sinusoidal':
      return SinusoidalPositionalEncoding(d_model, max_len)
    else:
      raise ValueError(f"Unsupported pos_type: {pos_type}")
  
  def forward(
      self, 
      x: torch.Tensor, 
      enc_output: torch.Tensor, 
      tgt_attn_mask: Optional[torch.Tensor] = None,
      tgt_key_padding_mask: Optional[torch.Tensor] = None,
      memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if self.pos is not None:
      x = self.pos(x)
    x = self.drop(x)

    for layer in self.layers:
      x = layer(
        x, 
        enc_output, 
        tgt_attn_mask=tgt_attn_mask, 
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask
      )
    x = self.final_ln(x)
    return x

class OriginalTransformer(nn.Module):
  def __init__(self, src_vocab_size: int, tgt_vocab_size: int, cfg: TransformerConfig, tie_weights: bool = False):
    super(OriginalTransformer, self).__init__()
    self.src_embed = TokenEmbedding(src_vocab_size, cfg.d_model)
    self.tgt_embed = TokenEmbedding(tgt_vocab_size, cfg.d_model)
    self.encoder = EncoderBackbone(cfg)
    self.decoder = DecoderBackbone(cfg)
    self.output_proj = nn.Linear(cfg.d_model, tgt_vocab_size, bias=False)
    if tie_weights:
      self.output_proj.weight = self.tgt_embed.embedding.weight

  def forward(
      self, 
      src: torch.Tensor, 
      tgt: torch.Tensor, 
      src_key_padding_mask: Optional[torch.Tensor] = None, 
      tgt_key_padding_mask: Optional[torch.Tensor] = None, 
      tgt_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if tgt_attn_mask is None:
      seq_len = tgt.size(1)
      device = tgt.device
      tgt_attn_mask = mask_causal_mask(seq_len, device)
    enc_output = self.encoder(self.src_embed(src), key_padding_mask=src_key_padding_mask)
    dec_output = self.decoder(
      self.tgt_embed(tgt), 
      enc_output, 
      tgt_attn_mask=tgt_attn_mask, 
      tgt_key_padding_mask=tgt_key_padding_mask,
      memory_key_padding_mask=src_key_padding_mask
    )
    output = self.output_proj(dec_output)
    return output

def _sanity_check():
  torch.manual_seed(0)

  cfg = TransformerConfig(
    d_model=128,
    num_heads=4,
    d_ff=512,
    num_layers=2,
    dropout=0.1,
    pre_ln=False,
    pos_type='sinusoidal',
    max_len=128
  )

  model = OriginalTransformer(src_vocab_size=1000, tgt_vocab_size=1000, cfg=cfg, tie_weights=False)
  src = torch.randint(0, 1000, (2, 10)) # (batch_size=2, src_seq_len=10)
  tgt = torch.randint(0, 1000, (2, 10)) # (batch_size=2, tgt_seq_len=10)
  src_key_padding_mask = torch.zeros((2, 10), dtype=torch.bool) # (batch_size=2, src_seq_len=10)
  tgt_key_padding_mask = torch.zeros((2, 10), dtype=torch.bool) # (batch_size=2, tgt_seq_len=10)

  logits = model(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
  assert logits.shape == (2, 10, 1000), f"Expected output shape (2, 10, 1000) but got {logits.shape}"
  print("Sanity check passed!")

if __name__ == "__main__":
  _sanity_check()