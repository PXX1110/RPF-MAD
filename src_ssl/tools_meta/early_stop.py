import os
import torch
import numpy as np

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None      #None
        self.early_stop = False
        self.test_acc_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, test_acc, epoch, max_epoch, phase, test_acc_ci95, test_loss, model, optimizer, 
                 loss_rec, acc_rec, log_dir, log_file_path, i, loader_len, attack_num):
        score = -test_acc
        # score = test_acc
        # DSR = (test_acc/opt.cca)*100

        if self.best_score is None:
            self.best_score = score
            if phase == "train":
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} ± {:.2f} % '\
                  .format(epoch, max_epoch, i, loader_len,  test_loss, test_acc, test_acc_ci95))
            elif phase == "val":
                log(log_file_path, 'Validation Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} ± {:.2f} % '\
                  .format(epoch, max_epoch, i, loader_len,  test_loss, test_acc, test_acc_ci95))
            elif phase == "test":
                self.save_checkpoint(test_acc, model, optimizer, loss_rec, acc_rec, log_dir, i, attack_num)
                log(log_file_path, 'Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} ± {:.2f} % '\
                  .format(i, loader_len,  test_loss, test_acc, test_acc_ci95))
        # elif score < self.best_score + self.delta:
        elif score >= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                log(log_file_path, 'Best episode:{:0>3}'.format(i))
        else:
            self.best_score = score
            if phase == "train":
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} ± {:.2f} % '\
                  .format(epoch, max_epoch, i, loader_len,  test_loss, test_acc, test_acc_ci95))
            elif phase == "val":
                log(log_file_path, 'Validation Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} ± {:.2f} % '\
                  .format(epoch, max_epoch, i, loader_len,  test_loss, test_acc, test_acc_ci95))
            elif phase == "test":
                self.save_checkpoint(test_acc, model, optimizer, loss_rec, acc_rec, log_dir, i, attack_num)
                log(log_file_path, 'Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} ± {:.2f} % '\
                  .format(i, loader_len,  test_loss, test_acc, test_acc_ci95))    
            self.counter = 0
    
    def save_checkpoint(self, test_acc, model, optimizer, loss_rec, acc_rec, log_dir, i, attack_num):
        # '''Saves model when validation loss decrease.'''
        '''Saves model when validation acc increase.'''
        if self.verbose:
            self.trace_func(f'Validation acc increased ({self.test_acc_min:.6f} --> {test_acc:.6f}).  Saving model ...')
        checkpoint = {"state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_rec": loss_rec,
            "acc_rec": acc_rec,
            "episode": i}
        path_checkpoint = os.path.join(log_dir,"./best_MAT_checkpint_attack_{}.pkl".format(attack_num))
        torch.save(checkpoint, path_checkpoint)
        self.test_acc_min = test_acc
