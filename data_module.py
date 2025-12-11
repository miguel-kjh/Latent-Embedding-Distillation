# data_module.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset
from itertools import chain
import os

class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        batch_size: int = 8,
        max_length: int = 2048,
        patch_size: int = 4,
        num_workers: int = 4,
        dataset_config_name: str = None,
        train_file: str = None,
        validation_file: str = None,
        text_column_name: str = "text",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = None
        
    def setup(self, stage=None):
        # 1. Cargar Tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)
            # Asegurar tokens especiales
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Cargar Dataset
        if self.hparams.train_file and self.hparams.validation_file:
            data_files = {"train": self.hparams.train_file, "validation": self.hparams.validation_file}
            extension = self.hparams.train_file.split(".")[-1]
            if extension == "txt": extension = "text"
            if extension == "jsonl": extension = "json"
            dataset = load_dataset(extension, data_files=data_files)
        else:
            dataset = load_dataset(self.hparams.dataset_name, self.hparams.dataset_config_name)

        # 3. Procesamiento (Tokenización y Grouping)
        # Esta es la parte crítica adaptada del script original
        block_size = self.hparams.max_length
        patch_size = self.hparams.patch_size
        text_col = self.hparams.text_column_name

        def tokenize_function(examples):
            return self.tokenizer(examples[text_col])

        def group_texts(examples):
            # Lógica extraída de train_autoencoder.py original
            # Concatena textos y asegura divisibilidad por patch_size
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            
            # Recortamos al múltiplo más cercano de block_size que también respete patch_size
            # (Aunque block_size debería ser múltiplo de patch_size por diseño)
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            
            # Dividir en chunks
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            
            # Asegurar padding para el último chunk si fuera necesario (raro tras el recorte, pero por seguridad)
            # En el script original hacían un padding manual complejo. 
            # Aquí, como recortamos el dataset concatenado, garantizamos divisibilidad.
            # Solo verificamos que block_size sea divisible por patch_size.
            return result

        assert block_size % patch_size == 0, f"Block size ({block_size}) must be divisible by patch size ({patch_size})"

        # Aplicar transformaciones
        with self.tokenizer.deprecation_warnings(False): # Silenciar warnings molestos
            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=[text_col],
                desc="Tokenizing dataset"
            )

            self.lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                desc=f"Grouping texts in chunks of {block_size}"
            )

    def train_dataloader(self):
        return DataLoader(
            self.lm_datasets["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.data_collator
        )

    def val_dataloader(self):
        if "validation" not in self.lm_datasets:
            return None
        return DataLoader(
            self.lm_datasets["validation"],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.data_collator
        )

    def data_collator(self, features):
        # Collate simple porque ya tenemos bloques de tamaño fijo
        import torch
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        return batch