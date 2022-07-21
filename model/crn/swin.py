import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x):
        return self.fn(self.LayerNorm(x))


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.ffn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def window_partition(x, window_size):
    '''
    Args:
        x: b x t x d_model
            - b: batch size
            - t: total input size (#patches)
            - d_model: embedding dimension
        window_size: local window size (int)
    Do:
        b x t x d_model -> n x w x d_model
            - w: local window size
            - n: b * #windows (b * (t/w))
    '''

    assert len(x.shape) == 3
    b, t, d = x.shape
    x = x.view(-1, window_size, d)
    
    return x


def window_reverse(x, batch_size):
    '''
    Args:
        x: n x w x d_model
            - n: b * #windows (b * (t/w))
            - w: local window size
            - d_model: embedding dimension
        window_size: local window size (int)
    Do:
        n x w x d_model -> b x t x d_model
            - b: batch size
            - t: total input size (#patches)
    '''

    assert len(x.shape) == 3
    x = rearrange(x, "(b n) w d -> b (n w) d", b=batch_size)

    return x


class ShotEmbedding(nn.Module): 
    def __init__(
        self,
        nn_size,
        input_dim,
        d_model,
        hidden_dropout_prob=0.1
    ):
        super().__init__()

        '''
        Args:
            nn_size: 2*K (16)
            input_dim: 2048 (from shot encoder)
            d_model: 768 (embedding dimension -> d_model)
            hidden_dropout_prob: 0.1
        Do: 
            b x 2K x 2048(input_dim) -> b x 2K x d_model
                - b: batch size
                - K: initial neighbor size
                - d_model: embedding dimension
            + positional embedding
        '''

        self.shot_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Embedding(nn_size, d_model)
        self.register_buffer("pos_ids", torch.arange(nn_size, dtype=torch.long))
        # self.mask_embedding = -> 안 필요할듯 MSM에서 사용하는 것 같음

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        shot_emb = self.shot_embedding(x) # b x 2K x d_model
        pos_emb = self.position_embedding(self.pos_ids) # 2K x d_model
        embeddings = shot_emb + pos_emb # b x 2K x d_model
        embeddings = self.dropout(self.LayerNorm(embeddings)) # b x 2K x d_model

        return embeddings


class PatchMerging(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.LayerNorm = nn.LayerNorm(2*d_model)
        self.reduction = nn.Linear(2*d_model, d_model, bias=False)

    def forward(self, x):
        '''
        Args:
            x: b x t x d_model
                - b: batch size
                - t: total input size (#patches)
                - d_model: embedding dimension
        Do:
            b x t x d_model -> b x t' x d_model
            where t' = t/2
        '''

        x = rearrange(x, "b (p t) d -> b t (p d)", p=2) # b x t x d_model -> b x (t/2) x 2*d_model
        x = self.LayerNorm(x) # b x (t/2) x 2*d_model
        x = self.reduction(x) # b x (t/2) x 2*d_model -> b x (t/2) x d_model

        return x


def create_mask(window_size):
    mask = torch.ones(window_size, window_size) # w x w
    assert window_size % 2 == 0
    displacement = window_size // 2

    # mask out quadrand-1,-3
    mask[:displacement, -displacement:] = 0
    mask[-displacement:, :displacement] = 0

    return mask.to('cuda')


def scaled_dot_product(q, k, v, mask=None):
    '''
    Args:
        q: Q @ W_q (n x h x w x d_k)
        k: K @ W_k (n x h x w x d_k)
        v: V @ W_v (n x h x w x d_k)
            n: batch size * #windows
            h: #heads
            w: window size
            d_k: hidden dimension
        mask: window_size x window_size
    Do:
        (Q @ K^T / sqrt(d_K)) @ V 
        n x h x w x d_k -> n x h x w x d_k
    '''

    d_k = q.size()[-1]
    attention = torch.matmul(q, k.transpose(-2, -1)) # n x h x w x w
    attention = attention / math.sqrt(d_k) # n x h x w x w
    if mask is not None:
        attention = attention.masked_fill(mask==0, -9e15)
    attention = F.softmax(attention, dim=-1)
    values = torch.matmul(attention, v) # n x h x w x d_k

    return values


class MultiheadAttention(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads,
        batch_size, 
        window_size, 
        shifted
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.window_size = window_size
        self.displacement = window_size // 2
        self.shifted = shifted

        self.w_qkv = nn.Linear(self.d_model, 3*d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        '''
        Args:
            x: b x t x d_model
                - b: batch size
                - t: total input size (#patches)
                - d_model: embedding dimension
        Do:
            multi-head self-attention layer
            b x t x d_model -> b x t x d_model
        '''

        if self.shifted:
            x = torch.roll(x, shifts=-self.displacement, dims=1)

        # b x t x d_model -> n x w x d_model
        x = window_partition(x, self.window_size)

        n, w, d = x.shape
        qkv = self.w_qkv(x) # n x w x 3*d_model

        # separate Q, K, V
        qkv = rearrange(qkv, "n w (h d_k qkv) -> n h w (d_k qkv)", h=self.num_heads, qkv=3) # n x h x w x 3*d_k
        q, k, v = qkv.chunk(3, dim=-1) # n x h x w x d_k

        # calculate values
        # n x h x w x d_k
        if self.shifted:
            values = scaled_dot_product(q, k, v, create_mask(self.window_size))
        else:
            values = scaled_dot_product(q, k, v)

        values = rearrange(values, "n h w d_k -> n w (h d_k)") # n x w x d_model
        values = self.w_o(values) # n x w x d_model

        # n x w x d_model -> b x t x d_model
        x = window_reverse(x, self.batch_size)

        if self.shifted:
            x = torch.roll(x, shifts=self.displacement, dims=1)

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        batch_size,
        window_size,
        shifted
    ):
        super().__init__()
        
        self.attention_block = Residual(PreNorm(d_model, MultiheadAttention(d_model=d_model,
                                                                            num_heads=num_heads,
                                                                            batch_size=batch_size,
                                                                            window_size=window_size,
                                                                            shifted=shifted)))
        self.mlp_block = Residual(PreNorm(d_model, FeedForward(input_dim=d_model, hidden_dim=d_model)))

    def forward(self, x):
        '''
        Args:
            x: b x t x d_model
        Do:
            attention block -> mlp block
        '''

        x = self.attention_block(x)
        x = self.mlp_block(x)

        return x


class StageModule(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        batch_size,
        window_size
    ):
        super().__init__()
        assert num_layers % 2 == 0, "The number of layers in a stage must be divisible by 2"
        
        self.layers = nn.ModuleList([])
        for _ in range(num_layers // 2):
            self.layers.append(nn.ModuleList([
                SwinTransformerBlock(d_model=d_model, num_heads=num_heads, batch_size=batch_size,
                                     window_size=window_size, shifted=False), # W-MSA
                SwinTransformerBlock(d_model=d_model, num_heads=num_heads, batch_size=batch_size,
                                     window_size=window_size, shifted=True)   # SW-MSA
            ]))

    def forward(self, x):
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)

        return x


class SwinTransformerCRN(nn.Module):
    def __init__(
        self,
        nn_size,
        num_layers,
        input_dim,
        d_model,
        num_heads,
        batch_size,
        window_size,
        hidden_dropout_prob
    ):
        super().__init__()
        
        self.stage1 = StageModule(num_layers=num_layers[0], d_model=d_model, num_heads=num_heads[0],
                                   batch_size=batch_size, window_size=window_size[0])
        self.stage2 = StageModule(num_layers=num_layers[1], d_model=d_model, num_heads=num_heads[1],
                                   batch_size=batch_size, window_size=window_size[1])

        self.shot_embedding = ShotEmbedding(
                                nn_size=nn_size,
                                input_dim=input_dim,
                                d_model=d_model,
                                hidden_dropout_prob=hidden_dropout_prob)
        self.patch_merging = PatchMerging(d_model=d_model)

        '''
        self.classification_head = nn.Sequential(
            nn.LayerNorm(4*d_model),
            nn.Linear(4*d_model, 2)
        )
        '''

    def forward(self, x):
        x = self.shot_embedding(x) # b x 2K x d_model
        x = self.stage1(x) # b x 2K x d_model
        x = self.patch_merging(x) # b x K x d_model
        x = self.stage2(x) # b x K x d_model
        x = self.patch_merging(x) # b x (K/2) x d_model
        #x = rearrange(x, "b t d -> b (t d)") # b x (K/2)*d_model (flatten)
        #x = self.classification_head(x) # b x 2

        return x