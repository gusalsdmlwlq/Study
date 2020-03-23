import torch
import numpy as np
import argparse
import os


def make_dataset():
    dataset = []
    for i in range(16):
        x = np.random.rand(3)
        y = np.random.randint(low=0, high=3)
        dataset.append((x,y))
    return dataset

def custom_collate(batch):
    batch_size = len(batch)
    inputs = []
    labels = []
    for b in batch:
        x, y = b
        inputs.append(x)
        labels.append(y)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
torch.backends.cudnn.benchmark = True

dataset = make_dataset()
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = torch.utils.data.DataLoader(dataset, batch_size =4, sampler=sampler, collate_fn=custom_collate)

model = torch.nn.Linear(3,3).cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

for batch in loader:
    inputs, labels = batch
    outputs = model(inputs.cuda())
    loss = torch.nn.functional.cross_entropy(outputs, labels.cuda())
    print("GPU: {}, output: {}, loss: {}".format(args.local_rank, outputs, loss))