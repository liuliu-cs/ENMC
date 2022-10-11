# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools

import torch
import torch.nn as nn

import seq2seq.data.config as config
from seq2seq.models.attention import BahdanauAttention
from seq2seq.utils import init_lstm_
import seq2seq.models.dre as dre
from seq2seq.models.quantize import quantize
import torch.nn.functional as F
import numpy as np
import hnswlib
from sitq import Mips

class RecurrentAttention(nn.Module):
    """
    LSTM wrapped with an attention module.
    """
    def __init__(self, input_size=1024, context_size=1024, hidden_size=1024,
                 num_layers=1, batch_first=False, dropout=0.2,
                 init_weight=0.1):
        """
        Constructor for the RecurrentAttention.

        :param input_size: number of features in input tensor
        :param context_size: number of features in output from encoder
        :param hidden_size: internal hidden size
        :param num_layers: number of layers in LSTM
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param dropout: probability of dropout (on input to LSTM layer)
        :param init_weight: range for the uniform initializer
        """

        super(RecurrentAttention, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bias=True,
                           batch_first=batch_first)
        init_lstm_(self.rnn, init_weight)

        self.attn = BahdanauAttention(hidden_size, context_size, context_size,
                                      normalize=True, batch_first=batch_first)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, context, context_len):
        """
        Execute RecurrentAttention.

        :param inputs: tensor with inputs
        :param hidden: hidden state for LSTM layer
        :param context: context tensor from encoder
        :param context_len: vector of encoder sequence lengths

        :returns (rnn_outputs, hidden, attn_output, attn_scores)
        """
        # set attention mask, sequences have different lengths, this mask
        # allows to include only valid elements of context in attention's
        # softmax
        self.attn.set_mask(context_len, context)

        inputs = self.dropout(inputs)
        rnn_outputs, hidden = self.rnn(inputs, hidden)
        attn_outputs, scores = self.attn(rnn_outputs, context)

        return rnn_outputs, hidden, attn_outputs, scores


class Classifier(nn.Module):
    """
    Fully-connected classifier
    """
    def __init__(self, in_features, out_features, init_weight=0.1):
        """
        Constructor for the Classifier.

        :param in_features: number of input features
        :param out_features: number of output features (size of vocabulary)
        :param init_weight: range for the uniform initializer
        """
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.classifier.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.classifier.bias.data, -init_weight, init_weight)

    def forward(self, x):
        """
        Execute the classifier.

        :param x: output from decoder
        """
        out = self.classifier(x)
        return out


class ResidualRecurrentDecoder(nn.Module):
    """
    Decoder with Embedding, LSTM layers, attention, residual connections and
    optinal dropout.

    Attention implemented in this module is different than the attention
    discussed in the GNMT arxiv paper. In this model the output from the first
    LSTM layer of the decoder goes into the attention module, then the
    re-weighted context is concatenated with inputs to all subsequent LSTM
    layers in the decoder at the current timestep.

    Residual connections are enabled after 3rd LSTM layer, dropout is applied
    on inputs to LSTM layers.
    """
    def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2,
                 batch_first=False, embedder=None, init_weight=0.1):
        """
        Constructor of the ResidualRecurrentDecoder.

        :param vocab_size: size of vocabulary
        :param hidden_size: hidden size for LSMT layers
        :param num_layers: number of LSTM layers
        :param dropout: probability of dropout (on input to LSTM layers)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param embedder: instance of nn.Embedding, if None constructor will
            create new embedding layer
        :param init_weight: range for the uniform initializer
        """
        super(ResidualRecurrentDecoder, self).__init__()

        self.num_layers = num_layers

        self.att_rnn = RecurrentAttention(hidden_size, hidden_size,
                                          hidden_size, num_layers=1,
                                          batch_first=batch_first,
                                          dropout=dropout)

        self.rnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.rnn_layers.append(
                nn.LSTM(2 * hidden_size, hidden_size, num_layers=1, bias=True,
                        batch_first=batch_first))

        for lstm in self.rnn_layers:
            init_lstm_(lstm, init_weight)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size,
                                         padding_idx=config.PAD)
            nn.init.uniform_(self.embedder.weight.data, -init_weight,
                             init_weight)

        self.classifier = Classifier(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

        print('Classifier W: ', self.classifier.classifier.weight.shape)

        # for DRE
        self.enable_dre = False
        self.dre_inference = False
        self.qtize = None
        self.num_candidates = None
        self.w_shape = None
        self.profiling = False
        self.method = 'dre'

    def dre_init(self, scale=0.25, device='cpu', method='dre'):
        w = self.classifier.classifier
        w_shape = w.weight.size()
        self.w_shape = w_shape
        if method == 'dre':
            self.register_buffer('proj_matrix', dre.fit(
                w_shape[0], w_shape[1], scale=scale, dist='ternary'
            ).to(device))
            self.register_parameter('screen_w', nn.Parameter(
                dre.srp(w.weight.data, getattr(self, 'proj_matrix'))
            ))
            self.register_parameter('screen_b', nn.Parameter(
                w.bias.data
                # torch.zeros(w_shape[0])
            ))
        elif method in ['svd', 'knn', 'mips']:
            u, s, vh = np.linalg.svd(w.weight.data.numpy(), full_matrices=False)
            print('SVD: ', u.shape, s.shape, vh.shape)
            self.register_buffer('preview_w', torch.from_numpy(u @ np.diag(s)))
            self.register_buffer('vh', torch.from_numpy(vh))
            self.register_buffer('b', w.bias.data)

            if method in ['knn', 'mips']:
                W = w.weight.data.cpu().numpy()
                b = w.bias.data.cpu().numpy()
                b = np.expand_dims(b, axis=1)
                print('W: {}; b: {}'.format(W.shape, b.shape))
                Wb = np.concatenate((W, b), axis=1)
                n, d = W.shape[0], W.shape[1]
                
                if method == 'knn':
                    print('Using KNN search')
                    ef = 50
                    G = hnswlib.Index(space='ip', dim=d+1)
                    G.init_index(max_elements=n, ef_construction=200, M=16)
                    indices = np.arange(n)
                    G.add_items(Wb, indices)
                    G.set_ef(ef)
                    self.graph = G
                    print('Graph construction done, Wb: {}; ef: {}'.format(Wb.shape, ef))
                elif method == 'mips':
                    print('Using MIPS search')
                    self.mips = Mips(signature_size=8)
                    self.mips.fit(Wb)
                    print('MIPS construction done')            

    def dre_activate(self, enable_dre=False, threshold=None, dre_inference=False,
            qtize=None, num_candidates=None, preview=None, profiling=False, method='dre'):
        self.method = method
        self.enable_dre = enable_dre
        self.num_candidates = num_candidates
        self.preview = preview
        self.profiling = profiling
        if method == 'dre':
            self.dre_inference = dre_inference
            screen_w = getattr(self, 'screen_w')
            screen_w.requires_grad = enable_dre
            screen_b = getattr(self, 'screen_b')
            screen_b.requires_grad = enable_dre
            if qtize:
                print('Quantize sreen W to INT-{}'.format(qtize))
                screen_w.data = quantize(screen_w.data, num_bits=qtize, dequantize=True, signed=True)
                if self.profiling:
                    np.save('screener_weight', screen_w.data.cpu().numpy())
                    np.save('screener_bias', screen_b.data.cpu().numpy())
                self.qtize = qtize

    def init_hidden(self, hidden):
        """
        Converts flattened hidden state (from sequence generator) into a tuple
        of hidden states.

        :param hidden: None or flattened hidden state for decoder RNN layers
        """
        if hidden is not None:
            # per-layer chunks
            hidden = hidden.chunk(self.num_layers)
            # (h, c) chunks for LSTM layer
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * self.num_layers

        self.next_hidden = []
        return hidden

    def append_hidden(self, h):
        """
        Appends the hidden vector h to the list of internal hidden states.

        :param h: hidden vector
        """
        if self.inference:
            self.next_hidden.append(h)

    def package_hidden(self):
        """
        Flattens the hidden state from all LSTM layers into one tensor (for
        the sequence generator).
        """
        if self.inference:
            hidden = torch.cat(tuple(itertools.chain(*self.next_hidden)))
        else:
            hidden = None
        return hidden

    def forward(self, inputs, context, inference=False):
        """
        Execute the decoder.

        :param inputs: tensor with inputs to the decoder
        :param context: state of encoder, encoder sequence lengths and hidden
            state of decoder's LSTM layers
        :param inference: if True stores and repackages hidden state
        """
        self.inference = inference

        enc_context, enc_len, hidden = context
        hidden = self.init_hidden(hidden)

        x = self.embedder(inputs)

        x, h, attn, scores = self.att_rnn(x, hidden[0], enc_context, enc_len)
        self.append_hidden(h)

        x = torch.cat((x, attn), dim=2)
        x = self.dropout(x)
        x, h = self.rnn_layers[0](x, hidden[1])
        self.append_hidden(h)

        for i in range(1, len(self.rnn_layers)):
            residual = x
            x = torch.cat((x, attn), dim=2)
            x = self.dropout(x)
            x, h = self.rnn_layers[i](x, hidden[i + 1])
            self.append_hidden(h)
            x = x + residual

        if self.enable_dre:
            if self.method == 'dre':
                h = x.clone().detach()
                if self.qtize:
                    h = quantize(h, num_bits=self.qtize, dequantize=True, signed=True)
                    if self.profiling:
                        np.save('input_fixed4', h.cpu().numpy())
                h = dre.srp(h, getattr(self, 'proj_matrix'))
                z = F.linear(h, getattr(self, 'screen_w'), getattr(self, 'screen_b'))

                t = self.classifier(x)
                
                if self.dre_inference:
                    topk_v, _ = torch.topk(z, self.num_candidates, dim=-1)
                    topk_th = topk_v[:,:,-1].unsqueeze(-1).expand_as(z)
                    m = torch.gt(z, topk_th).to(torch.float)
                    if self.profiling:
                        print(x.shape)
                        print(z.shape)
                        print(t.shape)
                        print('num_candidates: ', self.num_candidates)
                        np.save('classifier_threshold', topk_th.cpu().numpy())
                    #     np.save('classifier_mask', m.cpu().numpy())
                        self.profiling = False
                    t = torch.mul(t, m) + torch.mul(z, 1.0 - m)
                    hidden = self.package_hidden()
                    return t, scores, [enc_context, enc_len, hidden] 

                hidden = self.package_hidden()
                return t, scores, [enc_context, enc_len, hidden], (t, z)
                # return t, scores, [enc_context, enc_len, hidden]
            elif self.method == 'svd':
                h = x.clone().detach()
                h = F.linear(h, getattr(self, 'vh'))
                pw = getattr(self, 'preview_w')
                z = F.linear(h[:,:,:128], pw[:,:128], getattr(self, 'b'))
                topk_v, _ = torch.topk(z, self.num_candidates, dim=-1)
                topk_th = topk_v[:,:,-1].unsqueeze(-1).expand_as(z)
                m = torch.gt(z, topk_th).to(torch.float)
                z_full = F.linear(h, pw, getattr(self, 'b'))
                z = torch.mul(z_full, m) + torch.mul(z, 1.0 - m)
                if self.profiling:
                    print(h.shape, pw.shape)
                    print(z.shape)
                    self.profiling = False
                hidden = self.package_hidden()
                return z, scores, [enc_context, enc_len, hidden]
            elif self.method in ['knn', 'mips']:
                # h = x.clone().detach()
                x = torch.squeeze(x)
                t = self.classifier(x)

                query = x.cpu().numpy()
                ones = np.ones((query.shape[0], 1))
                # print('query: {}, ones: {}'.format(query.shape, ones.shape))
                query = np.concatenate((query, ones), axis=1)
                if self.method == 'knn':
                    # KNN search for indices
                    I, _ = self.graph.knn_query(query, k=self.num_candidates)
                    m = torch.zeros(t.shape)
                    m.scatter_(-1, torch.from_numpy(I.astype(np.int64)), 1.)
                    m = m.to(torch.device('cuda:0'))
                elif self.method == 'mips':
                    # MIPS search for indices
                    M = list()
                    for q in query:
                        I, _ = self.mips.search(q, limit=self.num_candidates, require_items=True)
                        m = torch.zeros(self.w_shape[0])
                        m.scatter_(-1, torch.from_numpy(I), 1.)
                        M.append(m)
                    M = torch.stack(M)
                    M = torch.reshape(M, t.shape)
                    # print('M: ', M.shape)
                    m = M.to(torch.device('cuda:0'))

                # Using SVD approximation for unselected
                # x = F.linear(x, getattr(self, 'vh'))
                # pw = getattr(self, 'preview_w')
                # z_svd = F.linear(x[:,:self.preview], pw[:,:self.preview])
                z_inf = torch.full(t.shape, -10.).to(torch.device('cuda:0'))
                t_new = torch.mul(t, m) + torch.mul(z_inf, 1.0 - m)

                if self.profiling:
                    print('query: ', query.shape)
                    # print('KNN, I: ', I.shape)
                    print('m: ', m.shape)
                    _, topk_indices = torch.topk(t, 5, dim=-1)
                    print(topk_indices[-1])
                    print(I)
                    # o_mask = torch.zeros(t.shape)
                    # o_mask.scatter_(-1, topk_indices, 1.).to(torch.device('cuda:0'))
                    # print('missed: {:2f}'.format( torch.sum(torch.gt(o_mask - m, 1.)) / m.numel() ))
                    self.profiling = False

                hidden = self.package_hidden()
                return t_new, scores, [enc_context, enc_len, hidden] 
        else:
            x = self.classifier(x)
            hidden = self.package_hidden()

            return x, scores, [enc_context, enc_len, hidden]
