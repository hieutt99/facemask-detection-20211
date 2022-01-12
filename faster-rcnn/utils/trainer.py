import os, sys
from torch.optim import Adam, AdamW, lr_scheduler
import torch.functional as F
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch 
import json

from utils.arguments import TrainingArguments
import logging

logger = logging.getLogger(__name__)

logFormatter = logging.Formatter('[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.DEBUG)


GLOBAL_STATE_NAME = 'global_state.json'
MODEL_NAME = 'model.pt'
SCHEDULER_NAME = 'scheduler.pt'
OPTIMIZER_NAME = 'optimizer.pt'

def _set_device(dev):
    try:
        device = torch.device(dev)
        logger.info(f"Using {device}")
    except:
        logger.info("Failed to manually assigned device")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f"Using {device}")
    return device

class Trainer:
    def __init__(self, model, criterion, train_args):
        self.args = TrainingArguments(**train_args)

        self.device = _set_device(self.args.device)

        self.model = model
        self.model.to(self.device)

        self.criterion = criterion
        if self.criterion != None:
            self.criterion.to(self.device)

        # self.optimizer = AdamW(params=self.model.parameters(), lr=self.args.lr)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr,
                            momentum=0.9, weight_decay=0.0005)

        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, 264 - 1)

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
        # self.lr_scheduler = lr_scheduler.LinearLR(
        #                     self.optimizer, 
        #                     total_iters=3,
        #                     )

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=2)
        # self.lr_scheduler = lr_scheduler.ExponentialLR(
        #     self.optimizer, 
        #     gamma=0.95,
        # )
        self.writer = SummaryWriter(self.args.logging_dir)

        self.global_step = 0
        self.global_epoch = 0

        if self.args.checkpoint:
            self.load_trainer_state()
            self.load_checkpoint(optimizer=self.args.load_optimizer, lr_scheduler=self.args.load_lr_scheduler)

    def load_checkpoint(self, path=None, optimizer=True, lr_scheduler=True):
        if path == None:
            if self.args.checkpoint != None:
                path = os.path.join(self.args.save_folder, f'model_step_{self.args.checkpoint}')
            else:
                logger.error(f"No checkpoint {self.args.checkpoint} in {self.args.save_folder}.")
        if optimizer:
            self.optimizer.load_state_dict(torch.load(os.path.join(path, OPTIMIZER_NAME), map_location=self.device))
        if lr_scheduler:
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(path, SCHEDULER_NAME)))
        self.model.load_state_dict(torch.load(os.path.join(path, MODEL_NAME), map_location=self.device))
        logger.info(f'Loaded checkpoint {self.global_step} from {path}')
        
    def load_trainer_state(self, path=None):
        if not path:
            path = self.args.save_folder
        try:
            with open(os.path.join(path, GLOBAL_STATE_NAME), 'r') as fp:
                d = json.load(fp)
            for k, v in d.items():
                setattr(self, k, v)

            logger.info(f"Loaded trainer state form {path}")
        except:
            raise FileNotFoundError("Error loading file global state")

    def save_trainer_state(self, path=None):
        if not path:
            path = self.args.save_folder
        try:
            d = {
                'global_step':self.global_step,
                'global_epoch':self.global_epoch,
            }
            with open(os.path.join(path, GLOBAL_STATE_NAME), 'w') as fp:
                json.dump(d, fp)
                logger.info(f"Saved trainer state to {path}")
        except:
            raise TypeError("Error write file global state")
    
    def _save_checkpoint(self, ):
        if os.path.exists(self.args.save_folder):
            folder = os.path.join(os.path.join(self.args.save_folder, f'model_step_{self.global_step}'))
            if not os.path.exists(folder):
                os.makedirs(folder)

            torch.save(self.optimizer.state_dict(), os.path.join(folder, OPTIMIZER_NAME))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(folder, SCHEDULER_NAME))
            torch.save(self.model.state_dict(), os.path.join(folder, MODEL_NAME))
            logger.info(f'Saved checkpoint {self.global_step} to {folder}')

    # override ====================================
    def _handle_batch(self, batch):
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        return images, labels

    def _train_one_batch(self, batch):
        self.model.train()
        images, labels = self._handle_batch(batch)

        self.optimizer.zero_grad()
        outputs = self.model(images)

        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return outputs, loss.item()

    def _eval_one_batch(self, batch):
        self.model.eval()
        images, labels = self._handle_batch(batch)
        with torch.no_grad():
            outputs = self.model(images)

        return outputs, labels

    def _train_with_step(self, loader):
        self._total_imgs = self.args.n_steps
        progress_bar = tqdm(range(self.args.n_steps))
        progress_bar.set_description("Training in steps: ")
        iterator = iter(loader)
        for i in range(self.args.n_steps):
            batch = next(iterator)
            outputs, loss = self._train_one_batch(batch)

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss)
            self.writer.add_scalar('training_loss', loss, global_step=self.global_step)
            self.global_step += 1

            if self.global_step%self.args.lr_steps == 0:
                self.lr_scheduler.step()

            if self.global_step%self.args.save_interval == 0:
                self._save_checkpoint()
                self.save_trainer_state()

        self._save_checkpoint()
        self.save_trainer_state()
        
    def _train_with_epoch(self, loader):
        self.model.train()
        for epoch in range(self.args.n_epochs):
            self._total_imgs = len(loader.dataset)
            progress_bar = tqdm(range(len(loader)))
            progress_bar.set_description(f"Training epoch {epoch}: ")
            training_loss = 0
            n_steps = len(loader)
            if self.args.steps_per_epoch:
                n_steps = self.args.steps_per_epoch
            iterator = iter(loader)
            for step in range(n_steps):
                batch = next(iterator)
                outputs, loss = self._train_one_batch(batch)
                training_loss += loss

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss)

                self.writer.add_scalar('training_loss', loss, global_step=self.global_step)
                self.global_step += 1

                if self.global_step%self.args.lr_steps == 0:
                    self.lr_scheduler.step()

                if self.global_step%self.args.save_interval == 0 and self.args.save_strategy=='step':
                    self._save_checkpoint()
                    self.save_trainer_state()
            progress_bar.close()
            self.global_epoch += 1
            if self.global_epoch%self.args.save_interval == 0 and self.args.save_strategy == 'epoch':
                self._save_checkpoint()
                self.save_trainer_state()

        self._save_checkpoint()
        self.save_trainer_state()
    # =====================================================
                 
    def train(self, loader, val_loader=None):
        logger.info("***TRAIN***")
        if self.args.strategy == 'epoch':
            self._train_with_epoch(loader, val_loader=val_loader)
        elif self.args.strategy == 'step':
            self._train_with_step(loader)
        else:
            logger.error("Invalid strategy")

    def predict(self, loader):
        pass