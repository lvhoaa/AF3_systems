# This code uses Pytorch profiler to profile performance and memory profiler to get memory consumption 
# Code version with raw Pytorch (without Pytorch lightning & Fabric)

from __future__ import annotations
import torch
from torch.utils.data import Dataset
from random import randrange
from alphafold3_pytorch import (
    Alphafold3,
    AtomInput,
    DataLoader,
)
import sys
from tests.test_trainer import MockAtomDataset
from functools import wraps, partial
from pathlib import Path
from alphafold3_pytorch.alphafold3 import Alphafold3
from alphafold3_pytorch.attention import pad_at_dim
from typing import TypedDict, List, Callable
from alphafold3_pytorch.typing import (
    typecheck,
    beartype_isinstance,
)
from alphafold3_pytorch.inputs import (
    AtomInput,
    INPUT_TO_ATOM_TRANSFORM
)
import torch
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset, DataLoader as OrigDataLoader
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from ema_pytorch import EMA
# from lightning import Fabric
# from lightning.fabric.wrappers import _unwrap_objects
from shortuuid import uuid
from torch.autograd.profiler import record_function
import logging
import socket
from datetime import datetime, timedelta
import matplotlib
from thop import profile
from deepspeed.utils.timer import SynchronizedWallClockTimer 

# Memory profiler configs
logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

    # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html")

   # Construct the trace file -- profile compute performance 
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

# Class Trainer, tweaked from trainer.py 
def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch

@typecheck
def accum_dict(
    past_losses: dict | None,
    losses: dict,
    scale: float = 1.
):
    losses = {k: v * scale for k, v in losses.items()}

    if not exists(past_losses):
        return losses

    for loss_name in past_losses.keys():
        past_losses[loss_name] += losses.get(loss_name, 0.)

    return past_losses

# dataloader and collation fn

@typecheck
def collate_af3_inputs(
    inputs: List,
    int_pad_value = -1,
    map_input_fn: Callable | None = None

) -> AtomInput:

    if exists(map_input_fn):
        inputs = [map_input_fn(i) for i in inputs]

    # go through all the inputs
    # and for any that is not AtomInput, try to transform it with the registered input type to corresponding registered function

    atom_inputs = []

    for i in inputs:
        if beartype_isinstance(i, AtomInput):
            atom_inputs.append(i)
            continue

        maybe_to_atom_fn = INPUT_TO_ATOM_TRANSFORM.get(type(i), None)

        if not exists(maybe_to_atom_fn):
            raise TypeError(f'invalid input type {type(i)} being passed into Trainer that is not converted to AtomInput correctly')

        atom_inputs.append(maybe_to_atom_fn(i))

    # separate input dictionary into keys and values

    keys = atom_inputs[0].keys()
    atom_inputs = [i.values() for i in atom_inputs]

    outputs = []

    for grouped in zip(*atom_inputs):
        # if all None, just return None

        not_none_grouped = [*filter(exists, grouped)]

        if len(not_none_grouped) == 0:
            outputs.append(None)
            continue

        # default to empty tensor for any Nones

        one_tensor = not_none_grouped[0]

        dtype = one_tensor.dtype
        ndim = one_tensor.ndim

        # use -1 for padding int values, for assuming int are labels - if not, handle within alphafold3

        if dtype in (torch.int, torch.long):
            pad_value = int_pad_value
        elif dtype == torch.bool:
            pad_value = False
        else:
            pad_value = 0.

        # get the max lengths across all dimensions

        shapes_as_tensor = torch.stack([Tensor(tuple(g.shape) if exists(g) else ((0,) * ndim)).int() for g in grouped], dim = -1)

        max_lengths = shapes_as_tensor.amax(dim = -1)

        default_tensor = torch.full(max_lengths.tolist(), pad_value, dtype = dtype)

        # pad across all dimensions

        padded_inputs = []

        for inp in grouped:

            if not exists(inp):
                padded_inputs.append(default_tensor)
                continue

            for dim, max_length in enumerate(max_lengths.tolist()):
                inp = pad_at_dim(inp, (0, max_length - inp.shape[dim]), value = pad_value, dim = dim)

            padded_inputs.append(inp)

        # stack

        stacked = torch.stack(padded_inputs)

        outputs.append(stacked)

    # reconstitute dictionary

    batched_atom_inputs = AtomInput(tuple(zip(keys, outputs)))
    return batched_atom_inputs

@typecheck
def DataLoader(
    *args,
    map_input_fn: Callable | None = None,
    **kwargs
):
    collate_fn = collate_af3_inputs

    if exists(map_input_fn):
        collate_fn = partial(collate_fn, map_input_fn = map_input_fn)

    return OrigDataLoader(*args, collate_fn = collate_fn, **kwargs)

# default scheduler used in paper w/ warmup

def default_lambda_lr_fn(steps):
    # 1000 step warmup

    if steps < 1000:
        return steps / 1000

    # decay 0.95 every 5e4 steps

    steps -= 1000
    return 0.95 ** (steps / 5e4)

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]

class Trainer:
    """ Section 5.4 """

    @typecheck
    def __init__(
        self,
        model: Alphafold3,
        *,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int = 1,
        map_dataset_input_fn: Callable | None = None,
        valid_dataset: Dataset | None = None,
        valid_every: int = 1000,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        ema_decay = 0.999,
        lr = 1.8e-3,
        default_adam_kwargs: dict = dict(
            betas = (0.9, 0.95),
            eps = 1e-8
        ),
        clip_grad_norm = 10.,
        default_lambda_lr = default_lambda_lr_fn,
        # fabric: Fabric | None = None,
        accelerator = 'auto',
        checkpoint_prefix = 'af3.ckpt.',
        checkpoint_every: int = 1000,
        checkpoint_folder: str = './checkpoints',
        overwrite_checkpoints: bool = False,
        # fabric_kwargs: dict = dict(),
        ema_kwargs: dict = dict(
            use_foreach = True
        )
    ):
        super().__init__()

        # if not exists(fabric):
        #     fabric = Fabric(accelerator = accelerator, **fabric_kwargs)

        # self.fabric = fabric
        # fabric.launch()

        # model

        self.model = model

        # exponential moving average

        # if self.is_main:
        #     self.ema_model = EMA(
        #         model,
        #         beta = ema_decay,
        #         include_online_model = False,
        #         **ema_kwargs
        #     )

        # optimizer

        if not exists(optimizer):
            optimizer = Adam(
                model.parameters(),
                lr = lr,
                **default_adam_kwargs
            )

        self.optimizer = optimizer

        # if map dataset function given, curry into DataLoader

        DataLoader_ = DataLoader

        if exists(map_dataset_input_fn):
            DataLoader_ = partial(DataLoader_, map_input_fn = map_dataset_input_fn)

        # train dataloader
        self.dataloader = DataLoader_(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        # training steps and num gradient accum steps

        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every

        # setup fabric

        # self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

        # fabric.setup_dataloaders(self.dataloader)

        # scheduler

        if not exists(scheduler):
            scheduler = LambdaLR(optimizer, lr_lambda = default_lambda_lr)

        self.scheduler = scheduler

        # gradient clipping norm

        self.clip_grad_norm = clip_grad_norm

        # steps

        self.steps = 0

        # checkpointing logic

        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_every = checkpoint_every
        self.overwrite_checkpoints = overwrite_checkpoints
        self.checkpoint_folder = Path(checkpoint_folder)

        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        # save the path for the last loaded model, if any

        self.train_id = None

        self.last_loaded_train_id = None
        self.model_loaded_from_path: Path | None = None

    # @property
    # def is_main(self):
        # return self.fabric.global_rank == 0

    def generate_train_id(self):
        if exists(self.train_id):
            return

        self.train_id = uuid()[:4].lower()

    @property
    def train_id_with_prev(self) -> str:
        if not exists(self.last_loaded_train_id):
            return self.train_id

        ckpt_num = str(self.model_loaded_from_path).split('.')[-2]

        return f'{self.last_loaded_train_id}.{ckpt_num}-{self.train_id}'

    # saving and loading

    def save_checkpoint(self):
        assert exists(self.train_id_with_prev)

        # formulate checkpoint path and save

        checkpoint_path = self.checkpoint_folder / f'({self.train_id_with_prev})_{self.checkpoint_prefix}{self.steps}.pt'

        self.save(checkpoint_path, overwrite = self.overwrite_checkpoints)

    def save(
        self,
        path: str | Path,
        overwrite = False,
        prefix: str | None = None
    ):
        if isinstance(path, str):
            path = Path(path)

        assert not path.is_dir() and (not path.exists() or overwrite)

        path.parent.mkdir(exist_ok = True, parents = True)

        unwrapped_model = _unwrap_objects(self.model)
        unwrapped_optimizer = _unwrap_objects(self.optimizer)

        package = dict(
            model = unwrapped_model.state_dict_with_init_args,
            optimizer = unwrapped_optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            steps = self.steps,
            id = self.train_id
        )

        torch.save(package, str(path))

    def load_from_checkpoint_folder(
        self,
        **kwargs
    ):
        self.load(path = self.checkpoint_folder, **kwargs)

    def load(
        self,
        path: str | Path,
        strict = True,
        prefix = None,
        only_model = False,
        reset_steps = False
    ):
        if isinstance(path, str):
            path = Path(path)

        assert path.exists(), f'{str(path)} cannot be found for loading'

        # if the path is a directory, then automatically load latest checkpoint

        if path.is_dir():
            prefix = default(prefix, self.checkpoint_prefix)

            model_paths = [*path.glob(f'**/*_{prefix}*.pt')]

            assert len(model_paths) > 0, f'no files found in directory {path}'

            model_paths = sorted(model_paths, key = lambda p: int(str(p).split('.')[-2]))

            path = model_paths[-1]

        # get unwrapped model and optimizer

        unwrapped_model = _unwrap_objects(self.model)

        # load model from path

        model_id = unwrapped_model.load(path)

        # for eventually saving entire training history in filename

        self.model_loaded_from_path = path
        self.last_loaded_train_id = model_id

        if only_model:
            return

        # load optimizer and scheduler states

        package = torch.load(str(path))

        unwrapped_optimizer = _unwrap_objects(self.optimizer)

        if 'optimizer' in package:
            unwrapped_optimizer.load_state_dict(package['optimizer'])

        if 'scheduler' in package:
            self.scheduler.load_state_dict(package['scheduler'])

        if reset_steps:
            self.steps = 0
        else:
            self.steps = package.get('steps', 0)

    # shortcut methods

    # def wait(self):
    #     self.fabric.barrier()

    def print(self, *args, **kwargs):
        # self.fabric.print(*args, **kwargs)
        print(*args, **kwargs)

    # def log(self, **log_data):
    #     self.fabric.log_dict(log_data, step = self.steps)
        

    # main train forwards

    def __call__(self):
        
        
        self.generate_train_id()

        # cycle through dataloader
        dl = cycle(self.dataloader)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=10, active=5, repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            # iterate training steps
            while self.steps < self.num_train_steps:
                print('Training step: ', self.steps)
                
                prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
                
                self.model.train()
                
                # gradient accumulation
                total_loss = 0.
                train_loss_breakdown = None

                for grad_accum_step in range(self.grad_accum_every):
                    is_accumulating = grad_accum_step < (self.grad_accum_every - 1)
                    inputs = next(dl)
                    inputs = move_to_device(inputs, 'cuda') # move single batch to gpu                      
                                        
                    # with self.fabric.no_backward_sync(self.model, enabled = is_accumulating):
                    with record_function("## forward ##"):
                        # model forwards
                        loss, loss_breakdown = self.model(
                            molecule_atom_lens = inputs["molecule_atom_lens"],
                            atom_inputs = inputs["atom_inputs"],
                            atompair_inputs = inputs["atompair_inputs"],
                            molecule_ids = inputs["molecule_ids"],
                            additional_molecule_feats = inputs["additional_molecule_feats"],
                            msa = inputs["msa"],
                            msa_mask = inputs["msa_mask"],
                            templates = inputs["templates"],
                            template_mask = inputs["template_mask"],
                            atom_pos = inputs["atom_pos"],
                            molecule_atom_indices = inputs["molecule_atom_indices"],
                            distance_labels = inputs["distance_labels"],
                            pae_labels = inputs["pae_labels"],
                            pde_labels = inputs["pde_labels"],
                            plddt_labels = inputs["plddt_labels"],
                            resolved_labels = inputs["resolved_labels"],
                            return_loss_breakdown = True
                        )
                        print("Memory usage after forward pass: ", SynchronizedWallClockTimer.memory_usage())
                                                
                        # multiply the sum of all parameters with zero and add it to the final loss
                        # loss += sum(torch.sum(param) for param in self.model.parameters()) * 0

                        # accumulate
                        scale = self.grad_accum_every ** -1
                        total_loss += loss.item() * scale
                        train_loss_breakdown = accum_dict(train_loss_breakdown, loss_breakdown._asdict(), scale = scale)                           
                        
                    with record_function("## backward ##"):
                        # backwards
                        # self.fabric.backward(loss / self.grad_accum_every)
                        (loss / self.grad_accum_every).backward()
                        print("Memory usage after backward pass: ", SynchronizedWallClockTimer.memory_usage())

                # log entire loss breakdown
                # self.log(**train_loss_breakdown)
                self.print(f'loss: {total_loss:.3f}')

                # clip gradients
                # self.fabric.clip_gradients(self.model, self.optimizer, max_norm = self.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), max_norm = self.clip_grad_norm)
                
                with record_function("## optimizer ##"):

                    # optimizer step
                    self.optimizer.step()
                    
                    # update exponential moving average
                    # self.wait()
                    # if self.is_main:
                    #     self.ema_model.update()
                    # self.wait()

                    # scheduler
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    print("Memory usage after optimizer step zero grad: ", SynchronizedWallClockTimer.memory_usage())
                    
                self.steps += 1
        print('training complete')

def calculate_model_size(alphafold3, dataset):
    # Measuring MACs and #Parameters
    # Use thop to measure # params and MACs 
    print('Using THOP')
    alphafold3.to('cuda')
    dl = cycle(DataLoader(dataset, batch_size = 1, shuffle = True, drop_last = True))
    inputs = next(dl)
    inputs = move_to_device(inputs, 'cuda')
    macs, params, thopDict = profile(model =alphafold3, inputs=inputs, ret_layer_info = True)
    print("Model size: macs", macs, " Params: ", params)
    
    # param.numel()
    print('Using param.numel()')
    total_params = 0 
    numelDict = {} 
    for name, param in alphafold3.named_parameters():
        val = param.numel()
        numelDict[name] = val 
        total_params += val
    print("Total params of model: ", total_params)
    
    # preprocessing numelDict
    toBeDeleted = [] 
    toBeAdded = []
    for key in numelDict:
        if key[-6:] == "weight" and key[0:-6] + "bias" in numelDict:
            prefix = key[0:-7]
            toBeAdded.append((prefix, numelDict[key] + numelDict[key[0:-6] + "bias"]))
            toBeDeleted.append(key)
            toBeDeleted.append(key[0:-6] + "bias")
        elif key[-6:] == "weight":
            toBeDeleted.append(key)
            toBeAdded.append((key[0:-7], numelDict[key]))
    for key in toBeDeleted:
        del numelDict[key]
    for key, val in toBeAdded:
        numelDict[key] = val
    
    # preprocessing thopDict -> thopValues
    # dfs helper function
    def dfs(currDict, prefix, storeDict):
        for key in currDict:
            if len(currDict[key][2]) == 0:
                # end dfs
                # Note: 0 is total_ops; 1 is total_params
                # Currently calculating total_ops
                storeDict[prefix + key] = currDict[key][0]
            else:
                # continue push dfs
                dfs(currDict[key][2], prefix + key + ".", storeDict)
    
    thopValues = {}
    dfs(thopDict, "", thopValues)
    
    print('thopValues: ', thopValues)
    print("===================================================")

    # Log out operator differences:
    #listKeys = set(numelDict.keys()).union(set(thopValues.keys()))
    #for key in listKeys:
    #    if numelDict.get(key, 0) != thopValues.get(key, 0):
    #        print('operator: ', key, '; numelValue: ', numelDict.get(key, 0), '; THOPValue: ', thopValues.get(key, 0))
    

def demo():
    # Full version of AlphaFold3 
    alphafold3 = Alphafold3(dim_atom_inputs = 77,dim_template_feats = 44)
    
    # Mini version: reduce all blocks depth to only 1 
    # alphafold3 = Alphafold3(
    #     dim_atom_inputs = 77,
    #     dim_template_feats = 44,
    #     num_dist_bins = 38,
    #     confidence_head_kwargs = dict(
    #         pairformer_depth = 1
    #     ),
    #     template_embedder_kwargs = dict(
    #         pairformer_stack_depth = 1
    #     ),
    #     msa_module_kwargs = dict(
    #         depth = 1
    #     ),
    #     pairformer_stack = dict(
    #         depth = 1
    #     ),
    #     diffusion_module_kwargs = dict(
    #         atom_encoder_depth = 1,
    #         token_transformer_depth = 1,
    #         atom_decoder_depth = 1,
    #     ),
    # )
    
    alphafold3.to('cuda')
    print("Memory usage after model initialization: ", SynchronizedWallClockTimer.memory_usage())
    
    dataset = MockAtomDataset(8)
    
    trainer = Trainer(
        model = alphafold3,
        dataset = dataset,
        num_train_steps = 15,
        batch_size = 1,
        valid_every = 1,
        grad_accum_every = 1,
        checkpoint_every = 1,
        overwrite_checkpoints = True,
        ema_kwargs = dict(
            use_foreach = True,
            update_after_step = 0,
            update_every = 1
        )
    )

    trainer()
    
    # calculate_model_size(alphafold3, dataset)
    
        
if __name__ == "__main__":
    print('Starting to run demo')
    demo()
    print('Finished demo')