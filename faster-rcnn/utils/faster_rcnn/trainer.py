import os, sys 
from utils.trainer import Trainer
import torch 
from tqdm.auto import tqdm 
from torchvision.utils import save_image, draw_bounding_boxes
from yolov5_utils.metrics import *
import logging
logger = logging.getLogger(__name__)

logFormatter = logging.Formatter('[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.DEBUG)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def process_batch(detections, labels, iou_thresholds):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
        iou_thresholds: list iou thresholds from 0.5 -> 0.95
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iou_thresholds.shape[0], dtype=torch.bool, device=iou_thresholds.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iou_thresholds[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        # [label, detection, iou]
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iou_thresholds.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iou_thresholds

    return correct

class FasterRCNNTrainer(Trainer):
    def __init__(self, model , train_args, data_args, **kwargs):
        super(FasterRCNNTrainer, self).__init__(model=model, criterion=None, train_args=train_args)
        self.iou_thresholds = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.num_thresholds = self.iou_thresholds.numel()
        self.data_args = data_args

    def handle_preds(self, preds, targets):
        
        for index, img in enumerate(preds):
            lb = targets[index]
            boxes = img['boxes'].cpu()
            l = img['labels'].unsqueeze(1).cpu()
            conf = img['scores'].unsqueeze(1).cpu()

            p = torch.cat((boxes, conf, l), dim=1)

            x = lb['boxes'].cpu()
            y = lb['labels'].unsqueeze(1).cpu()
            z = torch.cat((y, x), dim=1)

            for i in range(self.data_args.num_classes):
                self.num_per_class[i]+=len(torch.where(y[:,0] == i))


            target_class = z[:, 0].tolist() if self.data_args.num_classes else [] 
            if len(preds) == 0:
                self.stats.append((torch.zeros(0, self.num_thresholds, dtype=torch.bool),
                                    torch.Tensor(), torch.Tensor(), target_class))

            correct = process_batch(p, z, self.iou_thresholds)
            self.stats.append((correct.cpu().squeeze(1), conf.cpu().squeeze(1), l.cpu().squeeze(1), target_class))
            self.seen+=1

    def _convert_boxes(self, boxes, w, h):
        '''
        convert yolo boxes to xyxy boxes
        '''
        batch_size = boxes.size(0)
        pos_w = [[0,2] for i in range(batch_size)]
        pos_h = [[1,3] for i in range(batch_size)]
        w_components = torch.gather(boxes, dim=1, index=torch.tensor(pos_w))*w
        h_components = torch.gather(boxes, dim=1, index=torch.tensor(pos_h))*h
        boxes[:,0] = w_components[:, 0] - w_components[:,1]/2
        boxes[:,1] = h_components[:, 0] - h_components[:,1]/2
        boxes[:,2] = w_components[:, 0] + w_components[:,1]
        boxes[:,3] = h_components[:, 0] + h_components[:,1]
        return boxes

    def _handle_batch(self, batch, eval=False):
        device = self.device if not eval else torch.device('cpu')
        imgs, labels, paths, _ = batch 
        b, c, w, h = imgs.size()
        imgs = [img.to(device)/255 for img in imgs]
        targets = []
        for i in range(len(imgs)):
            query = labels[(labels[:,0]==i).nonzero().squeeze(1)]
            _labels = query[:,1].long()
            _boxes = query[:,-4:]
            targets.append({
                'boxes': self._convert_boxes(_boxes, w, h).to(device) if _boxes.size(0)>0 else _boxes.to(device),
                'labels': _labels.to(device)+1,
            })
        return imgs, targets    

    def _train_one_batch(self, batch):
        self.model.train()
        imgs, targets = self._handle_batch(batch)

        outputs = self.model(imgs, targets)

        return outputs 

    def _eval_one_batch(self, batch):
        cpu_device = torch.device('cpu')
        self.model.eval()
        imgs, targets = self._handle_batch(batch, eval=True)

        imgs = [img.to(self.device) for img in imgs]
        with torch.no_grad():
            predictions = self.model(imgs)
        # boxes, labels, scores
        predictions = [{k: v.to(cpu_device) for k, v in t.items()} for t in predictions]
        
        self.handle_preds(predictions, targets)
        # return predictions, metrics 

    def eval(self, loader, plots=False, verbose=True):
        num_iter = len(loader)
        progress_bar = tqdm(range(num_iter))
        progress_bar.set_description("Eval: ")

        self.stats, self.ap, self.ap_class = [], [], []
        self.seen = 0
        self.num_per_class = [0] * self.data_args.num_classes
        for index, batch in enumerate(loader):
            self._eval_one_batch(batch)

            progress_bar.update(1)
        
        progress_bar.close()
        self.stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        boxes_per_class = np.bincount(self.stats[2].astype(np.int64), minlength=self.data_args.num_classes)
        ap50 = None

        if len(self.stats) and self.stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*self.stats, plot=plots, save_dir=self.args.save_folder,
                # names=names,
                )
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, wap50, map50, map = p.mean(), r.mean(), cal_weighted_ap(ap50), ap50.mean(), ap.mean()
            nt = np.bincount(self.stats[3].astype(np.int64), minlength=self.data_args.num_classes)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        print_format = '%20s' + '%11i' * 3 + '%11.3g' * 5  # print format
        temp_format = '%20s' + '%11s' * 3 + '%11s' * 5
        print(temp_format %("Class", "Images","Labels", "Boxes", "P", "R", "wAP@.5", "mAP@.5", "mAP@.5:.95"))
        print(print_format % ('all', self.seen, nt.sum(), sum(boxes_per_class), mp, mr, wap50, map50, map))

        # Print results per class
        if verbose and (self.data_args.num_classes < 50) and self.data_args.num_classes > 1 and len(self.stats):
            for i, c in enumerate(ap_class):
                print(print_format % (c, self.num_per_class[i], nt[c],
                                    boxes_per_class[i], p[i], r[i], ap50[i], ap50[i], ap[i]))
        progress_bar.close()
        return 

    def _train_with_epoch(self, loader, val_loader):
        for epoch in range(self.global_epoch, self.args.n_epochs):
            iter_num = len(loader)
            progress_bar = tqdm(range(iter_num))
            progress_bar.set_description(f"Training epoch {epoch}: ")
            n_steps = len(loader)
            if self.args.steps_per_epoch:
                n_steps = self.args.steps_per_epoch
            iterator = iter(loader)
            for step in range(n_steps):
                try:
                    batch = next(iterator)
                except:
                    iterator = iter(loader)
                    batch = next(iterator)

                loss_dict = self._train_one_batch(batch)
                losses = sum(loss for loss in loss_dict.values())
                postfix = {k:v.item() for k,v in loss_dict.items()}
                
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                progress_bar.update(1)
                progress_bar.set_postfix(postfix)
                
                for k, v in postfix.items():
                    self.writer.add_scalar(k, v, global_step=self.global_step)
                self.writer.add_scalar("losses", losses.item(), global_step=self.global_step)
                
                self.global_step+=1
                
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
            if val_loader:
                self.eval(val_loader, verbose=True)

        self._save_checkpoint()
        self.save_trainer_state()
