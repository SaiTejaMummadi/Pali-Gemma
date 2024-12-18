from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
            self, 
            hidden_size=768,
            intermediate_size = 3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens= num_image_tokens



class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2D(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride=self.patch_size,
            padding="valid", #No padding is added
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False,
        )
    
    def forward(self,pixel_values:torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape #[Batchsize, Channels, Height, Width]
        #Covolve the 'patchsize' kernel over the image with no overlapping patches
        #The output of the convolution -> [Batchsize, EmbedDim, Numpatches h, numpatches w]
        #Numpatch_h = height // patchsize and likewise for width
        patch_embeds = self.patch_embedding(pixel_values)
        #[Batchsize, embeddim, numpatchesh, numpatchesw] -> [Batchsize, embeddim, numpatches]
        #numpatches = numpatchesh * numpatchesw
        embeddings = patch_embeds.flatten(2)
        #we are transposing the last two dimensions
        #[Batchsize, embeddim, numpatches] -> [batchsize, numpatches, embeddim]
        embeddings = embeddings.transpose(1,2)
        #Addding position emeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)
        #[batchsize, numpatches, embeddim]
        return embeddings

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 #Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout
    
        self.k_proj == nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj == nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj == nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj == nn.Linear(self.embed_dim, self.embed_dim)


    def forward(
            self, 
            hidden_states: torch.Tensor,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        #hidden_sates: [Batchsize, numpatches, embeddim]
        batch_size, seq_len, _ = hidden_states.size()
        #query_states : [Batchsize, numpatches, embeddim]
        query_states = self.q_prpj(hidden_states)
        #key_states : [Batchsize, numpatches, embeddim]
        key_states = self.k_proj(hidden_states)
        #value_states : [Batchsize, numpatches, embeddim]
        value_states = self.v_proj(hidden_states)
        #Transformation to Multihead
        #[Batchsize, numpatches, embedim] 
        # - > [Batchsize, numpatches, num_heads, head dim] 
        # - > [Batchsize, num_heads, numpatches, head dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads,self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads,self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads,self.head_dim).transpose(1,2)

        #Calculating the attention using formula Q*k^T / sqrt(d_k)
        #query states = [batchsize, numheads, numpatches, headdim]
        #key states (after transposing) = [batchsize, numheads, headdim, numpatches]
        # attn_weights = [batchsize, numheads, numpatches, numpatches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3))*self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f"{attn_weights.size()}"
            )
        
        #Apply the softmax row-wise, attn_weights : [batchsize, numheads, numpatches, numpatches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(query_states.dtype)

        #Apply drpout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        #Multiple the attention weights by the value states
        attn_output = torch.matmul(attn_weights, value_states)        

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f"{attn_output.size()}"
            )


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        #[batchsize, numpatches, embeddim] -> [Batchsize, numpatches, intermediatesize]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        hidden_states = self.fc2(hidden_states)
        return hidden_states
    

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
            self, 
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        #residual: [Batchsize, numpatches, embeddim]
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.self_attn(hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.layer_norm2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values:torch.Tensor) -> torch.Tensor:
        #Pixel_values: [Batchsize, Channels, Height, Width] -> [Batchsize, Numpatches, Embeddings]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state



class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self,pixel_values) -> Tuple:
        #[Batchsize, channels, heights, width] -> [Batchsize, num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)