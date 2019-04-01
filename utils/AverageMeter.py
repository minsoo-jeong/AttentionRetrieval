class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val=0, avg=0, sum=0, count=0):
        self.reset()
        self.val = val
        self.avg = avg
        self.sum = sum
        self.count = count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def state_dict(self):
        return {'val': self.val, 'sum': self.sum, 'count': self.count, 'avg': self.avg}
