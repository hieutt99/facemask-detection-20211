import os, sys 
from ..trainer import Trainer 
import torch
from tqdm.auto import tqdm
from torchvision.utils import save_image
from .computeloss import YoloComputeLoss


class YoloTrainer(Trainer):
    def __init__(self, model, train_args):
        super(YoloTrainer, self).__init__(model=model, criterion=None, train_args=train_args)
        self.compute_loss = YoloComputeLoss(model)

    def _handle_batch(self, batch):
        images = batch['images'].to(self.device)
        labels = batch['labels'].to(self.device)
        return images, labels

    def _train_one_batch(self, batch):
        self.model.train()
        images, labels = self._handle_batch(batch)
        
        self.optimizer.zero_grad()
        outputs = self.model(images)

        loss, loss_items = self.compute_loss(outputs, labels)
        loss.backward()

        self.optimizer.step()
        return outputs, loss, loss_items

    def _eval_one_batch(self, batch):
        self.model.eval()
        images, labels = self._handle_batch(batch)

        with torch.no_grad():
            outputs = self.model(images)
            loss, loss_items = self.compute_loss(outputs, labels)

        return outputs, loss, loss_items

    def eval(self, loader):
        self.model.eval()
        self._total_imgs = len(loader.dataset)
        progress_bar = tqdm(range(len(loader)))
        progress_bar.set_description("Eval: ")
        all_loss = []
        for index, item in enumerate(loader):
            images = self._handle_batch(item['images'])
            num = images.size(0)
            ws = num/self._total_imgs
            with torch.no_grad():
                outputs = self.model(images)
                loss_outputs = self.model.loss_function(*outputs, M_N=ws)
                all_loss.append(loss_outputs)
            progress_bar.update(1)
        return all_loss

    def generate(self, loader, save_folder=None):
        self.model.eval()
        progress_bar = tqdm(range(len(loader)))
        progress_bar.set_description("Generate: ")
        for index, item in enumerate(loader):
            # images = self._handle_batch(item)
            images = item['images'].to(self.device)
            outputs = self.model(images)
            if save_folder:
                save_image(outputs, os.path.join(save_folder, f'{index}_output.png'))
                save_image(images, os.path.join(save_folder, f'{index}_input.png'))

            progress_bar.update(1)
                
        progress_bar.close()

    def _train_with_step(self, loader):
        self._total_imgs = self.args.n_steps
        progress_bar = tqdm(range(self.args.n_steps))
        progress_bar.set_description("Training in steps: ")
        iterator = iter(loader)
        for i in range(self.args.n_steps):
            batch = next(iterator)
            outputs, loss, loss_items = self._train_one_batch(batch)

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
                outputs, loss, loss_items = self._train_one_batch(batch)
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