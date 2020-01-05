import torch.utils.data as data
import numpy as np
# from app.dataloader import DataLoader
from dataloader import DataLoader

# from app.lm_rnn import RNN_Encoder, SequentialRNN, LinearDecoder
from lm_rnn import RNN_Encoder, SequentialRNN, LinearDecoder

class TextSeqDataset(data.Dataset):
    def __init__(self, x, y, backwards=False, sos=None, eos=None):
        self.x, self.y, self.backwards, self.sos, self.eos = x, y, backwards, sos, eos

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]  # we need to get y as array
        if self.backwards:
            x = list(reversed(x))
        if self.eos is not None:
            x = x + [self.eos]
        if self.sos is not None:
            x = [self.sos]+x
        return np.array(x), np.array(y)

    def __len__(self): return len(self.x)


class SeqDataLoader(DataLoader):
    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        res[1] = np.reshape(res[1], -1)  # reshape the labels to one sequence
        return res


def freeze_all_but(learner, n):
    c = learner.get_layer_groups()
    for l in c:
        set_trainable(l, False)
    set_trainable(c[n], True)


class MultiBatchSeqRNN(RNN_Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        input, labels = input
        for l in self.hidden:
            for h in l:
                h.data.zero_()
        raw_outputs, outputs = super().forward(input)
        return raw_outputs, outputs, labels

def get_rnn_seq_tagger(bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                       dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, linear_decoder_dp=0.1, tag_to_idx={}, tagset_size=8):
    rnn_enc = MultiBatchSeqRNN(n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                               dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop)

    return SequentialRNN(rnn_enc, LinearDecoder(n_class, emb_sz, linear_decoder_dp, tag_to_idx, tagset_size))
