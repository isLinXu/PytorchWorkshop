import os
import torch
from torch.nn.utils.rnn import pad_sequence
vocabPath = './char.txt'

def getBatch(data):
  x = pad_sequence([torch.tensor([1] + [(vocabIndex[t] if t in vocabSet else 0) for t in s] + [0], dtype=torch.long) for s in data])
  l = [len(s) + 1 for s in data]
  mask = torch.ones_like(x)
  for i, t in enumerate(l):
    mask[t:, i].fill_(0)
  return x, l, mask

def initial(path):
  global vocab, vocabSet, vocabIndex
  with open(path, 'r', encoding='utf-8') as f:
    vocab = ['', ''] + f.read().split('\0')
  vocabSet = set(vocab)
  vocabIndex = {}
  for i, w in enumerate(vocab):
    vocabIndex[w] = i
  return vocab

vocab = []
if os.path.exists(vocabPath):
  initial(vocabPath)
