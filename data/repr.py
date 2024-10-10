import torch
from opt_einsum import contract as einsum
import esm
from data.residue_constants import order2restype_with_mask

def get_pre_repr(seqs, model, alphabet, batch_converter, device="cuda:0"):

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    # data = [
    #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    #     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein3",  "K A <mask> I S Q"),
    # ]
    data = []
    for idx, seq in enumerate([seqs]):
        seq_string = ''.join([order2restype_with_mask[int(i)] for i in seq])
        data.append(("protein_"+str(idx), seq_string))

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)

    node_repr = results["representations"][33][:,1:-1,:]
    pair_repr = results['attentions'][:,33-1,:,1:-1,1:-1].permute(0,2,3,1)
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    # sequence_representations = []
    # for i, tokens_len in enumerate(batch_lens):
    #     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    # Look at the unsupervised self-attention map contact predictions
    # for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    #     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    return node_repr, pair_repr