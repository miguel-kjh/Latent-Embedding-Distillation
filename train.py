# train.py
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from configuration_autoencoder import AutoencoderConfig
from modeling_vae import CALMAutoencoder
from data_module import TextDataModule

def main(args):
    pl.seed_everything(args.seed)

    # 1. Preparar Datos
    data_module = TextDataModule(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        patch_size=args.patch_size,
        num_workers=args.num_workers
    )
    
    # Llamamos a setup() manualmente para cargar el tokenizer y obtener vocab_size
    data_module.setup()
    vocab_size = len(data_module.tokenizer)
    print(f"Vocab size detected: {vocab_size}")

    # 2. Configurar Modelo
    config = AutoencoderConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        latent_size=args.latent_size,
        patch_size=args.patch_size,
        ae_dropout=args.ae_dropout,
        kl_weight=args.kl_weight,
        pad_token_id=data_module.tokenizer.pad_token_id,
        bos_token_id=data_module.tokenizer.bos_token_id,
        eos_token_id=data_module.tokenizer.eos_token_id,
    )

    model = CALMAutoencoder(config)

    # 3. Callbacks y Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='calm-ae-{epoch:02d}-{val_loss:.2f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Usar WandB si está instalado, sino TensorBoard
    try:
        logger = WandbLogger(project="calm-autoencoder", name=args.run_name)
    except:
        logger = TensorBoardLogger("tb_logs", name=args.run_name)

    # 4. Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",  # Usa todas las GPUs disponibles
        precision="bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 16,
        max_steps=args.max_steps,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    # 5. Entrenar
    print("Starting training...")
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar CALM Autoencoder con PyTorch Lightning")

    # Datos
    parser.add_argument("--dataset_name", type=str, default=None, help="Nombre del dataset en HF Hub")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Config del dataset")
    parser.add_argument("--train_file", type=str, default=None, help="Ruta a archivo de texto local")
    parser.add_argument("--validation_file", type=str, default=None, help="Ruta a archivo de validación local")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer base (ej. gpt2, llama)")
    
    # Hiperparámetros de Modelo
    parser.add_argument("--patch_size", type=int, default=4, help="K tokens a comprimir")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--ae_dropout", type=float, default=0.15)
    parser.add_argument("--kl_weight", type=float, default=1e-3)

    # Entrenamiento
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512, help="Tamaño del bloque de contexto")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default="calm-ae-run")

    args = parser.parse_args()
    
    # Validación básica
    if not args.dataset_name and not args.train_file:
        parser.error("Debes proporcionar --dataset_name o --train_file")

    main(args)