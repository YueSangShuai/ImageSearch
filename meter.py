import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        assert isinstance(val, float)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, labels):
    outputs = outputs.data.cpu().numpy()
    outputs = np.argmax(outputs, axis=1)
    labels = labels.data.cpu().numpy()
    acc = np.mean((outputs == labels).astype(int))
    return acc

class Meters():
    def __init__(self):
        self.meters = dict()
    def update(self, batchsize, **kwargs):
        for name, value in kwargs.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter()
            self.meters[name].update(float(value), batchsize)
    def reset(self):
        for name, value in self.meters.items():
            value.reset()
    def __str__(self) -> str:
        total = [(value.val, value.avg) for name, value in self.meters.items() if name!='acc']
        total = np.sum(np.array(total), axis=0)
        s = '| '.join([f'{name}: {value.val:.4f} {value.avg:.3f}' for name, value in self.meters.items()])
        if len(self.meters) > 2:
            return s + f'| total: {total[0]:.3f} {total[1]:.3f}'
        else:
            return s
    def __iter__(self):
        return iter(self.meters.items())
    def __getitem__(self, key):
        return self.meters[key]
    def items(self):
        return self.meters.items()
