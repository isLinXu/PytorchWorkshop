from .common import *
from .model import vocab
option = dict(edim=256, epochs=1.5, maxgrad=1., learningrate=1e-3, sdt_decay_step=1, batchsize=8, vocabsize=vocab, fp16=2, saveInterval=10, logInterval=.4)
option['loss'] = lambda opt, model, y, out, *_, rewards=[]: F.cross_entropy(out.transpose(-1, -2), y, reduction='none')
option['criterion'] = lambda y, out, mask, *_: (out[:,:,1:vocab].max(-1)[1] + 1).ne(y).float() * mask.float()
option['startEnv'] = lambda x, y, l, *args: (x, y, l, *args)
option['stepEnv'] = lambda i, pred, l, *args: (False, 1., None, None) # done episode, fake reward, Null next input, Null length, Null args
option['cumOut'] = False # True to keep trajectory
option['devices'] = [0] if torch.cuda.is_available() else [] # list of GPUs
option['init_method'] = 'file:///tmp/sharedfile' # initial configuration for multiple-GPU training
try:
    from qhoptim.pyt import QHAdam
    option['newOptimizer'] = lambda opt, params, _: QHAdam(params, lr=opt.learningrate, nus=(.7, .8), betas=(0.995, 0.999))
except ImportError: pass
