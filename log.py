import os

class Logger:
    def __init__(self, folder):
        self.folder = folder
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Create the progress.txt file
        self.log_file = os.path.join(folder, "progress.txt")
        with open(self.log_file, "a"):
            pass

    def log(self, text):
        with open(self.log_file, "a") as f:
            f.write(text + "\n")
            
    def log_kv(self, key, value):
     with open(self.log_file, "a") as f:
         f.write(f"{key}: {value}\n")