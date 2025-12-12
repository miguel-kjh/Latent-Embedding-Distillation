# configuration_autoencoder.py
from transformers import PretrainedConfig

class AutoencoderConfig(PretrainedConfig):
    model_type = "calm_autoencoder"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        intermediate_size: int = 2048,  # Ajustado a un valor más estándar para size 2048
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        latent_size: int = 128,
        patch_size: int = 4,
        ae_dropout: float = 0.15,
        kl_weight: float = 1e-3,
        kl_clamp: float = 0.5,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        embedding_layer_weights = None,
        **kwargs,
    ):
        """
        Configuración para el Autoencoder Variacional de CALM.
        
        Args:
            patch_size: Número de tokens a comprimir en un solo vector (K).
            ae_dropout: Probabilidad de dropout para el vector latente y tokens de entrada.
            kl_clamp: Valor mínimo para el recorte de la pérdida KL (evita colapso posterior).
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.ae_dropout = ae_dropout
        self.kl_weight = kl_weight
        self.kl_clamp = kl_clamp
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )