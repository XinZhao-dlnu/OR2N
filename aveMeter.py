class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.n = 0

    def update(self, val,n=1):
        self.val = val
        self.avg = (self.avg * self.n + val * n ) * 1.0 / (self.n + n )
        self.n = self.n + n
