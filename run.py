import torch
from alphafold3_pytorch import Alphafold3
from torch.optim import Adam
from torch.autograd.profiler import record_function

alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_template_feats = 44
)

# mock inputs
seq_len = 16
molecule_atom_lens = torch.randint(1, 3, (2, seq_len))
atom_seq_len = molecule_atom_lens.sum(dim = -1).amax()
atom_inputs = torch.randn(2, atom_seq_len, 77)
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5)
additional_molecule_feats = torch.randn(2, seq_len, 9)
molecule_ids = torch.randint(0, 32, (2, seq_len))
template_feats = torch.randn(2, 2, seq_len, seq_len, 44)
template_mask = torch.ones((2, 2)).bool()
msa = torch.randn(2, 7, seq_len, 64)
msa_mask = torch.ones((2, 7)).bool()
atom_pos = torch.randn(2, atom_seq_len, 3)
molecule_atom_indices = molecule_atom_lens - 1 # last atom, as an example
distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
pae_labels = torch.randint(0, 64, (2, seq_len, seq_len))
pde_labels = torch.randint(0, 64, (2, seq_len, seq_len))
plddt_labels = torch.randint(0, 50, (2, seq_len))
resolved_labels = torch.randint(0, 2, (2, seq_len))

optimizer = Adam(alphafold3.parameters(),lr = 0.001)

# train
def train(data):
    with record_function("## forward ##"):
        loss = alphafold3(
            num_recycling_steps = 2,
            atom_inputs = atom_inputs,
            atompair_inputs = atompair_inputs,
            molecule_ids = molecule_ids,
            molecule_atom_lens = molecule_atom_lens,
            additional_molecule_feats = additional_molecule_feats,
            msa = msa,
            msa_mask = msa_mask,
            templates = template_feats,
            template_mask = template_mask,
            atom_pos = atom_pos,
            molecule_atom_indices = molecule_atom_indices,
            distance_labels = distance_labels,
            pae_labels = pae_labels,
            pde_labels = pde_labels,
            plddt_labels = plddt_labels,
            resolved_labels = resolved_labels
        )
    with record_function("## backward ##"):
        loss.backward()
    with record_function("## optimizer ##"):
        optimizer.step()
        optimizer.zero_grad()
    
# MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 5000000 
# torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

with torch.profiler.profile(
    activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
    # schedule=torch.profiler.schedule(wait=1, warmup=11, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/fold'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(15):
        print('step: ', step)
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        train(None)
        
# try:
#     torch.cuda.memory._dump_snapshot(f"af3.pickle")
# except Exception as e:
#     print("Failed to capture memory snapshot", e)

# torch.cuda.memory._record_memory_history(enabled=None)