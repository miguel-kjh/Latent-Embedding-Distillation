# modeling_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Optional
from transformers.activations import ACT2FN
from configuration_autoencoder import AutoencoderConfig

# --- Componentes Auxiliares (Ligeros y sin dependencias externas) ---

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class MLP(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class AELayer(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        # Según el paper, el AE es context-free (procesa chunks independientemente),
        # por lo que usamos solo MLP, sin auto-atención[cite: 73, 76].
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        return residual + x

# --- Encoder y Decoder ---

class Encoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # El encoder se divide en dos etapas de procesamiento
        self.layers = nn.ModuleList([AELayer(config) for _ in range(config.num_encoder_layers)])
        self.num_stage_layers = config.num_encoder_layers // 2
        
        # Proyecciones
        self.squeeze_layer = nn.Linear(config.patch_size * config.hidden_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Salida: media y log_var (latent_size * 2)
        self.hidden_to_latent = nn.Linear(config.hidden_size, config.latent_size * 2)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # Input shaping: (Batch * Num_Patches, Patch_Size)
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # Procesamiento por etapas
        # Etapa 1: Procesar embeddings individuales
        for i in range(self.num_stage_layers):
            hidden_states = self.layers[i](hidden_states)

        # Compresión: aplanar patch_size embeddings en un vector
        # (B*N, K, H) -> (B*N, 1, K*H) -> (B*N, 1, H)
        hidden_states = hidden_states.view(hidden_states.shape[0], 1, -1)
        hidden_states = self.squeeze_layer(hidden_states)

        # Etapa 2: Procesar vector comprimido
        for i in range(self.num_stage_layers, len(self.layers)):
            hidden_states = self.layers[i](hidden_states)

        hidden_states = self.norm(hidden_states)
        latent_params = self.hidden_to_latent(hidden_states)
        return latent_params # (Batch * Num_Patches, 1, Latent*2)


class Decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.latent_to_hidden = nn.Linear(config.latent_size, config.hidden_size)
        
        self.layers = nn.ModuleList([AELayer(config) for _ in range(config.num_decoder_layers)])
        self.num_stage_layers = config.num_decoder_layers // 2
        
        self.expand_layer = nn.Linear(config.hidden_size, config.patch_size * config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # El head de salida compartirá pesos con el embedding del encoder
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.latent_to_hidden(latent_states)

        # Etapa 1: Procesar representación latente expandida
        for i in range(self.num_stage_layers):
            hidden_states = self.layers[i](hidden_states)

        # Expansión: (B*N, 1, H) -> (B*N, 1, K*H) -> (B*N, K, H)
        hidden_states = self.expand_layer(hidden_states)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], self.patch_size, -1)

        # Etapa 2: Procesar secuencia reconstruida
        for i in range(self.num_stage_layers, len(self.layers)):
            hidden_states = self.layers[i](hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

# --- Lightning Module Principal ---

class CALMAutoencoder(pl.LightningModule):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Tie weights (Compartir pesos embedding entrada/salida)
        if config.tie_word_embeddings:
            self.decoder.lm_head.weight = self.encoder.embed_tokens.weight

        # Inicialización de pesos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids):
        # input_ids shape esperado: (Batch, Sequence_Length)
        # Debe ser divisible por patch_size
        batch_size, seq_len = input_ids.shape
        assert seq_len % self.config.patch_size == 0, "Sequence length must be divisible by patch_size"
        
        # Reshape para procesar parches independientemente: (B * Num_Patches, Patch_Size)
        flat_input_ids = input_ids.reshape(-1, self.config.patch_size)
        
        # Encoding
        latent_params = self.encoder(flat_input_ids)
        mean, log_std = torch.chunk(latent_params, 2, dim=-1)
        
        # Modo inferencia: usamos la media (determinista)
        # Modo entrenamiento: usamos reparametrización (estocástico)
        if self.training:
            std = torch.exp(log_std)
            eps = torch.randn_like(mean)
            z = mean + eps * std
        else:
            z = mean
            
        # Decoding
        logits = self.decoder(z)
        
        # Reshape logits back to (Batch, Seq_Len, Vocab)
        logits = logits.reshape(batch_size, seq_len, -1)
        
        return logits, mean, log_std

    def _compute_loss(self, logits, targets, mean, log_std):
        # 1. Reconstruction Loss (Cross Entropy)
        # Flatten para compatibilidad con CE loss
        loss_fct = nn.CrossEntropyLoss()
        rec_loss = loss_fct(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        
        # 2. KL Divergence Loss con Clipping
        # KL(N(mu, sigma) || N(0, 1)) = 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2))
        std = torch.exp(log_std)
        kl_div = 0.5 * (torch.pow(mean, 2) + torch.pow(std, 2) - 1 - 2 * log_std)
        
        # "To mitigate this... we adopt the KL clipping strategy" [cite: 110]
        # Sumamos sobre dimensiones latentes
        kl_per_dim = torch.sum(kl_div, dim=-1) # (Batch*Patches, 1)
        # Aplicamos clipping por dimensión o promedio (El paper implica clipping por dimensión antes de sumar, 
        # pero la implementación original hace clamp sobre la suma o loss final. 
        # Aquí seguimos la implementación de referencia: clamp sobre el loss KL calculado)
        kl_loss = torch.mean(kl_per_dim)
        kl_loss = torch.clamp(kl_loss, min=self.config.kl_clamp)
        
        # Total Loss
        # "Total loss function is a weighted sum... L_total = L_rec + beta * L_KL" [cite: 99]
        # Nota: La implementación original escala la rec_loss por patch_size, mantenemos eso.
        total_loss = (rec_loss * self.config.patch_size) + (kl_loss * self.config.kl_weight)
        
        return total_loss, rec_loss, kl_loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"] # Asumiendo un dict del DataLoader
        
        # Preparamos inputs planos
        flat_input_ids = input_ids.reshape(-1, self.config.patch_size)
        targets = flat_input_ids.clone()
        
        # --- Input Token Dropout (Robustness) ---
        # "Randomly masking a fraction (p=0.15) of tokens" [cite: 119]
        if self.config.ae_dropout > 0:
            mask = torch.rand_like(flat_input_ids.float()) > self.config.ae_dropout
            # Multiplicamos por la máscara (asumiendo pad_id=0 es ignorado o manejado)
            # Ojo: esto asume que el token 0 no afecta gravemente al embedding si no es PAD
            masked_input_ids = flat_input_ids * mask.long()
        else:
            masked_input_ids = flat_input_ids

        # Forward Pass (Encoding)
        latent_params = self.encoder(masked_input_ids)
        mean, log_std = torch.chunk(latent_params, 2, dim=-1)
        
        # Reparameterization Trick
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        z = mean + eps * std
        
        # --- Latent Vector Dropout (Robustness) ---
        # "Dropout with a rate of p=0.15 to the latent vector z" [cite: 117]
        z = F.dropout(z, p=self.config.ae_dropout, training=True)
        
        # Decoding
        logits = self.decoder(z) # (B*N, K, Vocab)
        
        # Loss Calculation
        loss, rec_loss, kl_loss = self._compute_loss(logits, targets, mean, log_std)
        
        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/rec_loss", rec_loss)
        self.log("train/kl_loss", kl_loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        # En validación no aplicamos dropouts ni máscaras aleatorias usualmente,
        # pero para el VAE queremos medir la reconstrucción pura.
        
        logits, mean, log_std = self(input_ids)
        loss, rec_loss, kl_loss = self._compute_loss(logits, input_ids, mean, log_std)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rec_loss", rec_loss)
        
        # Token-level Accuracy (Metric simple)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == input_ids).float().mean()
        self.log("val/accuracy", acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        # Configuración estándar AdamW
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=1e-4, 
            weight_decay=0.1, 
            betas=(0.9, 0.95)
        )
        return optimizer