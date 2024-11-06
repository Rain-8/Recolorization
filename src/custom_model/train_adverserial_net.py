import os
from turtle import forward
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from accelerate import Accelerator
import wandb


class AdversarialTrainer:
    def __init__(self, encoder, decoder, discriminator, train_dataset, eval_dataset, args):
        # Initialize Accelerator
        self.accelerator = Accelerator()

        # Prepare model, optimizer, and dataloaders with accelerator
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.validation_interval = args.validation_interval
        self.logging_interval = args.logging_interval
        self.checkpointing_interval = args.checkpointing_interval
        self.checkpoint_dir = args.checkpoint_dir

        self.run_name = args.run_name
        if args.run_name is None:
            self.run_name = f"Adversarial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        wandb.init(project=args.project_name, name=self.run_name)

        # Initialize dataloaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.val_batch_size)

        # Optimizer and Scheduler
        self.optimizerRD = Adam(self.decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerD = Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.num_epochs = args.num_epochs
        self.num_training_steps = args.num_epochs * len(self.train_dataloader)
        self.criterion_MSE = nn.MSELoss()
        self.criterion_BSE = nn.BCELoss()
        
        for param in self.encoder.parameters(): 
            param.requires_grad = False

        # Prepare everything with Accelerator
        (
            self.encoder,
            self.decoder,
            self.discriminator,
            self.optimizerRD,
            self.optimizerD,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.encoder, self.decoder, self.discriminator, self.optimizerRD, self.optimizerD, self.train_dataloader, self.eval_dataloader
        )

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            progress_bar = tqdm(self.train_dataloader, disable=not self.accelerator.is_local_main_process)
            total_loss = 0.0
            for batch in progress_bar:
                # Forward pass
                src_image, tgt_image, illu, src_palette, tgt_palette = batch
                tgt_palette = tgt_palette.flatten()
                c1, c2, c3, c4 = self.encoder(src_image)
                outputs = self.decoder(c1, c2, c3, c4, tgt_palette)
                
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, tgt_image)
                self.accelerator.backward(loss)

                # Optimization step
                self.optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_dataloader)
            if epoch % self.logging_interval == 0:
                wandb.log({"epoch_train_loss": avg_train_loss, "epoch": epoch + 1})
            
            if epoch % self.checkpointing_interval == 0:
                self.save_checkpoint(epoch)

            # Evaluate at the end of each epoch
            if epoch % self.validation_interval == 0:
                self.evaluate()


