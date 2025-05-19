import torch.distributed as dist
import logging

def import_var(mname, class_name=None):
    import importlib
    if class_name is None:
        mname, class_name = mname.rsplit('.', 1)
    m = importlib.import_module(mname)
    c = getattr(m, class_name)
    return c

    
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def logging_info(msg):
    if is_main_process():
        logging.info(msg)