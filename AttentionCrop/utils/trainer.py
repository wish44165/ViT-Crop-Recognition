import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import get_cropping_model_loader
from utils.optimizer import create_optim
from utils.lr_scheduler import create_scheduler
from utils.model import CroppingModel

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = CroppingModel(cfg)
        self.optimizer = create_optim(cfg, self.model)
        self.lr_scheduler = create_scheduler(cfg, self.optimizer)
        self.dataloader_train, self.dataloader_val = get_cropping_model_loader(cfg)
        self.loss_function = nn.MSELoss()

        self.device = 'cuda'
        self.writer = SummaryWriter('{}/{}/tensorboard-info'.format(cfg['output']['output_folder'], cfg['output']['description']))

    def train(self):
        total_iterations = self.cfg['train']['total_iterations']
        train_iter = iter(self.dataloader_train)
        self.model = self.model.to(self.device)
        iteration_start = 0
        if self.cfg['keep_train']:
            iteration_start = self.load_model()+1
        progress_bar = tqdm(range(iteration_start, total_iterations+iteration_start), total=total_iterations+iteration_start, desc='Total iteration: ', initial=iteration_start)

        for iteration in progress_bar:
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(self.dataloader_train)
                data = next(train_iter)

            # Get the training data and move them to the target device
            img, label = (d.to(self.device) for d in data)
            
            # Forward propagation
            prediction = self.model(img)
            if len(prediction.shape) == 1:
                prediction = torch.unsqueeze(prediction, dim=0)
            loss = self.loss_function(prediction, label)

            # backward propagation & update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            ############# compute accuracy ###############
            # Make the shape of prediction to be BxHxW
            if len(prediction.shape) == 2:
                prediction = torch.unsqueeze(prediction, dim=0)
            # For every prediction in the batch
            batch_num_correct = 0
            batch_num_sample = 0
            for i in range(label.shape[0]):
                batch_num_correct += torch.eq((prediction[i] > 0), (label[i] > 0)).sum()
                batch_num_sample += torch.flatten(prediction).size()[0]
            accuracy = batch_num_correct / batch_num_sample

            self.writer.add_scalar('Accuracy/train', accuracy, iteration)
            self.writer.add_scalar('Loss/train', loss, iteration)

            if (iteration+1) % self.cfg['train']['eval_freq'] == 0:
                self.val(iteration+1)
            if (iteration+1) % self.cfg['train']['save_model_freq'] == 0:
                self.save_model(iteration+1)

    def val(self, iteration):
        total_iters_val = len(self.dataloader_val.dataset) // self.cfg['val']['batch_size']
        progress_bar = tqdm(enumerate(self.dataloader_val), total=total_iters_val, desc=f'Validating...')
        total_loss = 0.
        total_correct_num = 0.
        total_sample_num = 0
        with torch.no_grad():
            for i, data in progress_bar:
                # Get the training data and move them to the target device
                img, label = (d.to(self.device) for d in data)
            
                ############# Forward propagation ###############
                prediction = self.model(img)

                ############# Accumulate accuracy ###############
                # Make the shape of prediction to be BxHxW
                if len(prediction.shape) == 2:
                    prediction = torch.unsqueeze(prediction, dim=0)
                # For every prediction in the batch
                for i in range(label.shape[0]):
                    num_correct = torch.eq((prediction[i] > 0), (label[i] > 0)).sum()
                    num_sample = torch.flatten(prediction).size()[0]
                    total_correct_num += num_correct
                    total_sample_num += num_sample

                ############# Accumulate loss ###############
                loss = self.loss_function(prediction, label)
                total_loss += loss

            ############# Get loss and accuracy ###############
            total_loss /= total_iters_val
            accuracy = total_correct_num / total_sample_num
        print('=================================================')
        print(f'Validation result in {iteration}-th iteration:')
        print(f'Loss: {total_loss}')
        print(f'Accuracy: {accuracy}\n')
        self.writer.add_scalar('Loss/val', total_loss, iteration)
        self.writer.add_scalar('Accuracy/val', accuracy, iteration)

    def save_model(self, iteration):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'iteration': iteration
        }
        torch.save(checkpoint, "{}/{}/iteration_{}.pth".format(self.cfg['output']['output_folder'], self.cfg['output']['description'], iteration))

    def load_model(self):
        checkpoint = torch.load(self.cfg['path_pretrained_weight'])
        if 'model' in self.cfg['keep_train_obj']:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer' in self.cfg['keep_train_obj']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler' in self.cfg['keep_train_obj']:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['iteration']
