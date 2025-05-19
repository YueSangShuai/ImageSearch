from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        self.writer.add_image(tag, images, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins)
