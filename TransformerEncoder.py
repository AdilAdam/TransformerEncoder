import torch
import math
import torch.nn as nn
import torch.nn.functional as F



class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.emb(x)
        return x

class Posionalencoding(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 max_len: int=5000, 
                 reverse: bool=False, 
                 drop_rate: float =0.0) -> None:
        super(Posionalencoding, self).__init__()
        self.xscale = math.sqrt(d_model)
        self.pe = torch.zeros(max_len, d_model)
        
        self.drop_out = nn.Dropout(p=drop_rate)
        if reverse:
            position = torch.arange(max_len-1, -1, 1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0)/d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

        # self.register_buffer("pe", self.pe, persistent=False)
    def forward(self, x: torch.Tensor):
        
        self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        x = x * self.xscale + self.pe[:,:x.size(1)]
        return self.drop_out(x)


class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, tempereture: float, drop_rate: float=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = tempereture
        self.drop_out = nn.Dropout(p=drop_rate)

    def forward(self, q, k, v, mask = None)-> torch.Tensor:
        attn = torch.matmul(q/self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask==0,  -1e9)

        attn = self.drop_out(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class Multiheadattention(nn.Module):
    def __init__(self,
                d_model: int,
                n_heads: int,
                drop_rate :float=0.0):
        super().__init__()
        assert d_model%n_heads==0
        self.d_k=self.d_v = d_model//n_heads
        self.h = n_heads
        self.q_w = nn.Linear(d_model, self.h * self.d_k)
        self.k_w = nn.Linear(d_model, self.h * self.d_k)
        self.v_w = nn.Linear(d_model, self.h * self.d_v)
        self.o_w = nn.Linear(self.d_v * self.h, d_model)
        self.self_attn = ScaledDotProductAttention(tempereture= self.d_k ** 0.5, 
                                                   drop_rate=0.1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor, 
                mask: torch.Tensor =None)-> torch.Tensor:
        
        #q,k,v : [batch_size, sequence_len, d_model]
        residual = q
        n_batch, seq_len = q.size(0), q.size(1)
        q_w = self.q_w(q).view(n_batch, seq_len, self.h, self.d_k)
        k_w = self.k_w(k).view(n_batch, seq_len, self.h, self.d_k)
        v_w = self.v_w(v).view(n_batch, seq_len, self.h, self.d_k)
        q, k, v = q_w.transpose(1,2), k_w.transpose(1,2), v_w.transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        out, attn = self.self_attn(q=q, k=k, v=v, mask=mask) 
        out = out.transpose(1,2).contiguous().view(n_batch, seq_len, -1)
        out = self.dropout(self.o_w(out))
        out +=residual
        out = self.layer_norm(out)

        return out, attn


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.
    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 drop_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(drop_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:

        return self.w_2(self.dropout(self.activation(self.w_1(xs)))), xs


class EncoderBlock(nn.Module):
    def __init__(self,
                n_heads : int=4,
                d_model: int = 256,
                hidden_size: int = 512,
                drop_rate: float=0.5) -> None:
        super().__init__()
        self.attn = Multiheadattention(d_model,n_heads, drop_rate)
        self.ffw = PositionwiseFeedForward(d_model,hidden_size, drop_rate)
        
    def forward(self,x):
        out,_ = self.attn(q=x,k=x, v=x, mask=None)
        out, xs = self.ffw(out)
        return out, xs


class Encoder(nn.Module):
    def __init__(self, 
                 num_layer: int = 4,
                 vocab_size: int = 16,
                 n_heads: int = 4,
                 d_model: int = 256,
                 hidden_size: int =256,
                 drop_rate: float =0.5,

                 ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = Posionalencoding(d_model, drop_rate=drop_rate)
        self.layer = nn.ModuleList(EncoderBlock(n_heads, d_model, 
                                                hidden_size, drop_rate=drop_rate) for _ in range(num_layer))
       
    def forward(self, x):
        x = self.emb(x)
        x = self.pos(x)
        for layer in self.layer:
            out, xs = layer(x)
        return out, xs


class Transformer_Encoder(nn.Module):

    def __init__(self,
                num_layer=8,
                vocab_size=50,
                n_heads=8,
                d_model=512,
                hidden_size=1024,
                drop_rate=0.5,
                num_class: int= 4,
                 ):
        super().__init__()
        self.encoder = Encoder(num_layer,
                               vocab_size,
                               n_heads,
                               d_model,
                               hidden_size,
                               drop_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifer = nn.Linear(d_model, num_class)
        self.activ = nn.Sigmoid()
        
    def forward(self, x):
        ffw_x, xs = self.encoder(x)
        x = ffw_x + xs
        x =  self.layer_norm(x)
        x = self.classifer(x)
        return self.activ(x)

model = Transformer_Encoder()

# model.eval()
# model_size = sum(p.numel() for p in model.parameters()) / (1024*1024)

dict_ ={"2":1, "8":2, "14":3}

data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [15, 7, 9, 4, 12, 6, 2, 3, 8, 11, 14, 5, 13, 1, 10],
        [10, 15, 3, 8, 5, 6, 13, 12, 9, 1, 11, 14, 7, 4, 2]]

label = [[0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1]]


valid = []
model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
lossfn = nn.CrossEntropyLoss()
pred_logmax=nn.LogSoftmax(dim=1)
lr = 1e-4
optermizer = torch.optim.Adam(model.parameters(), lr=lr)
label = torch.tensor(label)

data = torch.tensor(data)
for i in range(2000):
    pred=model(data)
    pred = pred.transpose(2,1)
    loss = lossfn(pred, label)
    if i%100==0:
        print("loss: {:.3f}".format(loss.detach().float().tolist()))
        # if loss<=0.745:
        #     torch.save(model, "./mymodel{}.pth".format(i))
    optermizer.zero_grad()
    loss.backward()
    optermizer.step()