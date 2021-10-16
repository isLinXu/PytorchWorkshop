from .common import *
import sys
from pickle import HIGHEST_PROTOCOL
from time import time
from math import floor
import torch.optim as optim
from .data import newLoader
from .model import Model, predict
from .option import option
getNelement = lambda model: sum(map(lambda p: p.nelement(), model.parameters()))
l1Reg = lambda acc, cur: acc + cur.abs().sum(dtype=torch.float)
l2Reg = lambda acc, cur: acc + (cur * cur).sum(dtype=torch.float)
nan = torch.tensor(float('nan'), device=opt.device)
toDevice = lambda a, device: tuple(map(lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, a))
detach0 = lambda x: x[0].detach() if isinstance(x, torch.Tensor) else x[0]
modelPath = lambda epoch=None: 'model.epoch{}.pt'.format(epoch) if epoch else 'model.pt'
statePath = lambda epoch: 'train.epoch{}.pt'.format(epoch)
readModel = lambda model: model.module if hasattr(model, 'module') else model
saveModel = lambda model, path, hp=True: torch.save(readModel(model).state_dict(), path, pickle_protocol=HIGHEST_PROTOCOL if hp else 4)

def initParameters(opt, model):
  for m in model.modules():
    if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor):
      nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.PReLU):
      nn.init.constant_(next(m.parameters()), 1)
    if hasattr(m, '_reset_parameters'):
      m._reset_parameters()
  if opt.reset_parameters:
    opt.reset_parameters(opt, model)

def getParamOptions(opt, model, *config):
  res = []
  s = set()
  base = opt.learningrate
  for m, k in config:
    s = s.union(set(m.parameters()))
    res.append(dict(params=m.parameters(), lr=k * base))
  res.append({'params': filter(lambda p: not p in s, model.parameters())})
  return res

def step(opt, model, x, y, l, args, doPredict=True):
  args = toDevice(args, opt.device)
  x, y, l, *args = opt.startEnv(*toDevice((x, y), opt.device), l, *args)
  episode = True
  i, extraLoss = 0, 0
  out, others, rewards = [], [], []
  while episode:
    o, el, *os = model(x, *args)
    extraLoss += el
    if opt.cumOut:
      out.append(o)
      others.append(os)
    else:
      out = o
      others = os
    pred = predict(out, l, x, *args) if doPredict else None
    i += 1
    episode, reward, x, l, *args = opt.stepEnv(i, pred, l, *args)
    rewards.append(reward)
  return pred, rewards, y, out, extraLoss, others

def trainStep(opt, model, x, y, l, *otherData):
  optimizer = opt.optimizer
  optimizer.zero_grad()
  _, rewards, y, *out = step(opt, model, x, y, l, otherData, False)
  loss = opt.loss(opt, model, y, *out, rewards=rewards).sum(dtype=torch.float)
  if torch.allclose(loss, nan, equal_nan=True):
    raise Exception('Loss returns NaN')
  backward(loss, opt)
  if hasattr(opt, 'gradF'):
    opt.gradF(model, getParameters(opt, model))
  nn.utils.clip_grad_value_(getParameters(opt, model), opt.maxgrad)
  opt.optimizer.step()
  if hasattr(opt, 'paraF'):
    opt.paraF(opt, model)
  return float(loss)

def evaluateStep(opt, model, x, y, l, *args):
  pred, _, y, out, _, *others = step(opt, model, x, y, l, args)
  missed = opt.criterion(y, out, *toDevice(args, opt.device))
  return (float(missed.sum()), missed, pred, *others)

def evaluate(opt, model, path='val'):
  model.eval()
  totalErr = 0
  count = 0
  for x, y, l, *args in newLoader(path, opt, batch_size=opt.batchsize):
    count += len(l)
    err, _, pred, _, *others = evaluateStep(opt, model, x, y, l, *args)
    totalErr += err
  vs = tuple(map(detach0, others))
  if opt.drawVars:
    opt.drawVars(x[0], l[0], *vs)
    print(pred[0])
  return totalErr / count, opt.toImages(*vs) if opt.toImages else {}, count

def initTrain(opt, epoch=None, rank=0):
  model = Model(opt).train()
  if epoch:
    initParameters(opt, model)
    if type(epoch) == int:
      model.load_state_dict(torch.load(modelPath(epoch), map_location='cpu'))
  model = model.to(opt.device) # need before constructing optimizers
  paramOptions = getParamOptions(opt, model)
  eps = 1e-4 if opt.fp16 else 1e-8
  opt.optimizer = opt.newOptimizer(opt, paramOptions, eps)
  if opt.fp16:
    model, opt.optimizer = amp.initialize(model, opt.optimizer, opt_level="O{}".format(opt.fp16), **opt.ampArgs)
  if opt.sdt_decay_step > 0:
    gamma = opt.gamma if hasattr(opt, 'gamma') else .5
    opt.scheduler = optim.lr_scheduler.StepLR(opt.optimizer, opt.sdt_decay_step, gamma=gamma)
  else:
    opt.scheduler = None
  if type(epoch) == int and os.path.isfile(statePath(epoch)):
    state = torch.load(statePath(epoch), map_location='cpu')
    opt.optimizer.load_state_dict(state[0])
    if opt.scheduler:
      opt.scheduler.load_state_dict(state[1])
    if opt.fp16 and len(state) > 2:
      amp.load_state_dict(state[2])
  if opt.mp:
    if opt.cuda:
      from apex.parallel import DistributedDataParallel, convert_syncbn_model
      model = DistributedDataParallel(convert_syncbn_model(model), message_size=getNelement(model) - 1)
    else:
      from torch.nn.parallel import DistributedDataParallel
      model = DistributedDataParallel(model, device_ids=[opt.devices[rank]])
  if opt.cuda and rank == 0:
    print('GPU memory allocated before training: {} bytes'.format(torch.cuda.max_memory_allocated()))
    torch.cuda.reset_max_memory_allocated()
  return opt, model

def train(opt, model):
  last_epoch = opt.scheduler.last_epoch
  lastLog = last_epoch
  count = 0
  totalLoss = 0
  valErr = 0
  remainder = opt.epochs - floor(opt.epochs)
  epochs = int(opt.epochs) + (1 if remainder > 0 else 0)
  dataArgs = dict(path='train', opt=opt, batch_size=opt.batchsize, shuffle=True)
  try:
    from data import getIters
  except ImportError:
    getIters = lambda loader, **_: len(loader)
  start = time()
  for i in range(last_epoch, epochs):
    j = 0
    loader = newLoader(**dataArgs)
    iterLen = getIters(loader, **dataArgs)
    for x, y, l, *otherData in loader:
      count += len(l)
      profile = opt.profile and i == last_epoch and j == 0 and args.rank == 0
      with torch.autograd.profiler.profile(enabled=profile, use_cuda=opt.cuda) as prof:
        loss = trainStep(opt, model, x, y, l, *otherData)
        if profile:
          print(prof.key_averages().table())
          prof.export_chrome_trace('train-prof.trace')
      totalLoss += loss
      if opt.cuda and i == last_epoch and j == 0 and args.rank == 0:
        print('GPU memory usage of one minibatch: {} bytes'.format(torch.cuda.max_memory_allocated()))
      j += 1
      opt.currEpoch = i + j / iterLen
      if opt.currEpoch - lastLog >= opt.logInterval and args.rank == 0:
        torch.cuda.synchronize()
        valErr, vs, _ = evaluate(opt, model)
        avgLoss = totalLoss / count
        if opt.writer:
          opt.writer({'loss': avgLoss}, images=vs, histograms=dict(model.named_parameters()), n=opt.currEpoch)
        print('Epoch #{:.2f} | train loss: {:6.6f} | valid error: {:.4f} | learning rate: {:.5f} | train samples: {} | time elapsed: {:6.2f}s'
            .format(opt.currEpoch, avgLoss, valErr, opt.scheduler.get_last_lr()[0], count, time() - start))
        lastLog = opt.currEpoch
        totalLoss = 0
        count = 0
        model.train()
      if opt.currEpoch >= opt.epochs:
        break
    if opt.scheduler:
      opt.scheduler.step()
    if opt.saveInterval and (i + 1) % opt.saveInterval == 0 and args.rank == 0:
      torch.cuda.synchronize()
      saveState(opt, model, opt.scheduler.last_epoch)
  return valErr

def saveState(opt, model, epoch):
  modelName = modelPath(epoch)
  saveModel(model, modelName)
  state = [opt.optimizer.state_dict(), opt.scheduler.state_dict()]
  if opt.fp16:
    state.append(amp.state_dict())
  torch.save(state, statePath(epoch), pickle_protocol=HIGHEST_PROTOCOL)
  return modelName

def main(rank):
  global opt, model
  r0 = rank if rank >= 0 else 0
  args.rank = r0
  opt.device = torch.device('cuda:{}'.format(opt.devices[r0]))
  torch.cuda.set_device(opt.device)
  if rank < 0:
    opt.mp = False
  if opt.mp:
    print('Starting train process #{}'.format(rank))
    from torch.distributed import init_process_group
    init_process_group('nccl', world_size=nprocs, rank=rank, init_method=opt.init_method)
  torch.manual_seed(args.rank)
  np.random.seed(args.rank)
  opt, model = initTrain(opt, True, r0)
  if rank == 0:
    print('Number of parameters: {} | valid error: {:.3f}'.format(getNelement(model), evaluate(opt, model)[0]))
  if not '-init_only' in sys.argv and rank >= 0:
    train(opt, model)
    if rank == 0:
      err, _, count = evaluate(opt, model, 'test')
      print('Test error: {:.3f} | test samples: {}'.format(err, count))
  if rank == 0:
    modelName = saveState(opt, model, opt.scheduler.last_epoch if hasattr(opt, 'scheduler') else 0)
    print('Save model to {}'.format(modelName))

def cleanSharedFile(s):
  if s.startswith('file://'):
    path = s[7:]
    if os.path.isfile(path):
      os.unlink(s[7:])

try:
  from data import init
  init()
except ImportError: pass
opt.batchsize = 1
opt.epochs = 1
opt.maxgrad = 1. # max gradient
opt.dropout = 0
opt.learningrate = 0.001 # initial learning rate
opt.sdt_decay_step = 10 # how often to reduce learning rate
opt.criterion = lambda y, out, mask, *args: F.mse_loss(out, y) # criterion for evaluation
opt.loss = lambda opt, model, y, out, *args, **_: F.mse_loss(out, y) # criterion for loss function
opt.newOptimizer = lambda opt, params, eps: optim.Adam(params, lr=opt.learningrate, amsgrad=True, eps=eps)
opt.startEnv = lambda *args: args
opt.stepEnv = lambda *_: False, 1., None, None
opt.cumOut = False
opt.writer = 0 # TensorBoard writer
opt.drawVars = 0
opt.reset_parameters = 0
opt.toImages = 0
opt.profile = False
opt.saveInterval = 10
opt.logInterval = 1
opt.ampArgs = {}
opt.currEpoch = 0.
opt.devices = [0]
opt.__dict__.update(option)
nprocs = len(opt.devices)
opt.mp = nprocs > 1
opt.cuda &= torch.cuda.is_available()
if not opt.cuda:
  opt.fp16 = 0
if opt.cuda and opt.fp16 > 1:
  if not hasattr(opt, 'newOptimizer'):
    try:
      from apex.optimizers import FusedAdam
      FusedAdam([nan])
      opt.newOptimizer = lambda opt, params, _: FusedAdam(params, lr=opt.learningrate)
    except RuntimeError: pass
  getParameters = lambda opt, _: amp.master_params(opt.optimizer)
  def backward(loss, opt):
    with amp.scale_loss(loss, opt.optimizer) as scaled_loss:
      scaled_loss.backward()
else:
  getParameters = lambda _, model: model.parameters()
  backward = lambda loss, _: loss.backward()

if __name__ == '__main__':
  if opt.mp:
    cleanSharedFile(opt.init_method) 
    import torch.multiprocessing as mp
    print('Using GPUs {}.'.format(opt.devices))
    mp.spawn(main, nprocs=nprocs, join=True)
    main(-1)
    cleanSharedFile(opt.init_method)
  else:
    main(0)
