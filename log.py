import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        # Create the folder if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Create the progress.txt file
        self.log_file = os.path.join(log_dir, "progress.txt")
        with open(self.log_file, "a"):
            pass

    def log(self, text):
        print(text)
        with open(self.log_file, "a") as f:
            f.write(text + "\n")
            
    def log_kv(self, key, value):
     with open(self.log_file, "a") as f:
         f.write(f"{key}: {value}\n")
         
    def add_scalar(self, k, v, step):
        self.writer.add_scalar(k, v, step)
        
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)