from tqdm import tqdm
from datetime import datetime

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from accelerate import Accelerator
import wandb

class RecolorizeTrainer:
    def __init__(self, model, train_dataset, eval_dataset, args):
        # Initialize Accelerator
        self.accelerator = Accelerator()

        # Prepare model, optimizer, and dataloaders with accelerator
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.validation_interval = args.validation_interval
        self.logging_interval = args.logging_interval
        
        self.run_name = args.run_name
        if args.run_name is None:
            self.run_name = f"recolorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        wandb.init(project=args.project_name, name=self.run_name)

        # Initialize dataloaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.val_batch_size)

        # Optimizer and Scheduler
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        self.num_epochs = args.num_epochs
        self.num_training_steps = args.num_epochs * len(self.train_dataloader)
        
        # Prepare everything with Accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )

    def train(self):
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            progress_bar = tqdm(self.train_dataloader, disable=not self.accelerator.is_local_main_process)
            total_loss = 0.0
            for batch in progress_bar:
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                # Optimization step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_dataloader)
            if epoch % self.logging_interval == 0:
                wandb.log({"epoch_train_loss": avg_train_loss, "epoch": epoch + 1})
            
            # Evaluate at the end of each epoch
            if epoch % self.validation_interval == 0:
                self.evaluate()

    def evaluate(self):
        # Evaluation loop
        self.model.eval()
        total_loss = 0
        num_batches = len(self.eval_dataloader)
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Validation Loss: {avg_loss}")
