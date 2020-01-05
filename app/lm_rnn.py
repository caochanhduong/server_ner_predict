# -*- coding: utf-8 -*-
import warnings
from distutils.version import LooseVersion
# from .imports import *
# from .torch_imports import *
# from app.rnn_reg import LockedDropout,WeightDrop,EmbeddingDropout
from rnn_reg import LockedDropout,WeightDrop,EmbeddingDropout

# from model import Stepper
from core import set_grad_enabled
# from app.core import set_grad_enabled

from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')

# TODO: remove this later
START_TAG = "<START>"
STOP_TAG = "<STOP>"
# Compute log sum exp in a numerically stable way for the forward algorithm


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def seq2seq_reg(output, xtra, loss, alpha=0, beta=0):
    hs, dropped_hs = xtra
    if alpha:  # Activation Regularization
        loss = loss + (alpha * dropped_hs[-1].pow(2).mean()).sum()
    if beta:   # Temporal Activation Regularization (slowness)
        h = hs[-1]
        if len(h) > 1:
            loss = loss + (beta * (h[1:] - h[:-1]).pow(2).mean()).sum()
    return loss


def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    if IS_TORCH_04:
        return h.detach() if type(h) == torch.Tensor else tuple(repackage_var(v) for v in h)
    else:
        return Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)


class RNN_Encoder(nn.Module):

    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM or QRNN layers to drive the network, and
        - variational dropouts in the embedding and LSTM/QRNN layers

        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    initrange = 0.1

    def __init__(self, ntoken, emb_sz, n_hid, n_layers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5, qrnn=False):
        """ Default constructor for the RNN_Encoder class

            Args:
                bs (int): batch size of input data
                ntoken (int): number of vocabulary (or tokens) in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                n_hid (int): number of hidden activation per LSTM layer
                n_layers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropoute (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.

            Returns:
                None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs, self.qrnn = 1, qrnn
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        if self.qrnn:
            # Using QRNN requires cupy: https://github.com/cupy/cupy
            from .torchqrnn.qrnn import QRNNLayer
            self.rnns = [QRNNLayer(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                                   save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(n_layers)]
            if wdrop:
                for rnn in self.rnns:
                    rnn.linear = WeightDrop(
                        rnn.linear, wdrop, weights=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                                 1, bidirectional=bidir) for l in range(n_layers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.emb_sz, self.n_hid, self.n_layers, self.dropoute = emb_sz, n_hid, n_layers, dropoute
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList(
            [LockedDropout(dropouth) for l in range(n_layers)])

    def forward(self, input):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        sl, bs = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        with set_grad_enabled(self.training):
            emb = self.encoder_with_dropout(
                input, dropout=self.dropoute if self.training else 0)
            emb = self.dropouti(emb)
            raw_output = emb
            new_hidden, raw_outputs, outputs = [], [], []
            for l, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):
                current_input = raw_output
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.n_layers - 1:
                    raw_output = drop(raw_output)
                outputs.append(raw_output)

            self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        if IS_TORCH_04:
            return Variable(self.weights.new(self.ndir, self.bs, nh).zero_())
        else:
            return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)

    def reset(self):
        if self.qrnn:
            [r.reset() for r in self.rnns]
        self.weights = next(self.parameters()).data
        if self.qrnn:
            self.hidden = [self.one_hidden(l) for l in range(self.n_layers)]
        else:
            self.hidden = [(self.one_hidden(l), self.one_hidden(l))
                           for l in range(self.n_layers)]


class MultiBatchRNN(RNN_Encoder):
    def __init__(self, bptt, max_seq, *args, **kwargs):
        self.max_seq, self.bptt = max_seq, bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]

    def forward(self, input):
        sl, bs = input.size()
        for l in self.hidden:
            for h in l:
                h.data.zero_()
        raw_outputs, outputs = [], []
        for i in range(0, sl, self.bptt):
            r, o = super().forward(input[i: min(i+self.bptt, sl)])
            if i > (sl-self.max_seq):
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)


class LinearDecoder(nn.Module):
    initrange = 0.1

    def __init__(self, n_out, n_hid, dropout, tag_to_idx, tagset_size, tie_encoder=None, bias=False):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)
        self.tag_to_idx, self.tagset_size = tag_to_idx, tagset_size
        if bias:
            self.decoder.bias.data.zero_()
        if tie_encoder:
            self.decoder.weight = tie_encoder.weight

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        if torch.cuda.is_available():
            # Do the forward algorithm to compute the partition function
            init_alphas = torch.full((1, self.tagset_size), -10000.).cuda()
        else:
            init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_idx[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + \
            self.transitions[self.tag_to_idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _viterbi_decode(self, feats):
        backpointers = []

        if False:
            init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
        else:
            init_vvars = torch.full((1, self.tagset_size), -10000.)
        # Initialize the viterbi variables in log space

        init_vvars[0][self.tag_to_idx[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_idx[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        if torch.cuda.is_available():
            score = torch.zeros(1).cuda()
            tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long).cuda(), tags])
        else:
            score = torch.zeros(1)
            tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long), tags])
            
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
        return score

    def get_lstm_features(self, sentence):
        outputs = sentence
        output = self.dropout(outputs[-1])
        decoded = self.decoder(output.view(
            output.size(0)*output.size(1), output.size(2)))
        feats = decoded.view(-1, decoded.size(1))
        return feats

    def forward(self, input):
        _, outputs, tags = input
        feats = self.get_lstm_features(outputs)
        if (len(tags) > 0):
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq


class CustomCRF(nn.Module):
    def __init__(self, tag_to_idx, tagset_size):
        super().__init__()
        self.tag_to_idx = tag_to_idx
        self.tagset_size = tagset_size

        if torch.cuda.is_available():
            self.transitions = nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size)).cuda()
        else:
            self.transitions = nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        if torch.cuda.is_available():
            init_alphas = torch.full((1, self.tagset_size), -10000.).cuda()
        else:
            init_alphas = torch.full((1, self.tagset_size), -10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_idx[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + \
            self.transitions[self.tag_to_idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def forward(self, input):
        # NOTE: forward function return: transitions, feats (sentences_length, tagset_size)
        decoded, raw_outputs, outputs = input
        alpha = self._forward_alg(decoded)
        return self.transitions, self.tag_to_idx, self.tagset_size, decoded, alpha, raw_outputs, outputs


class LinearBlock(nn.Module):
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x): return self.lin(self.drop(self.bn(x)))


class PoolingLinearClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(bs, -1)

    def forward(self, input):
        raw_outputs, outputs = input
        output = outputs[-1]
        sl, bs, _ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x, raw_outputs, outputs


class SequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'):
                c.reset()


def get_rnn_classifier(bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                       dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, qrnn=False):
    rnn_enc = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                            dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)
    return SequentialRNN(rnn_enc, PoolingLinearClassifier(layers, drops))


get_rnn_classifer = get_rnn_classifier

