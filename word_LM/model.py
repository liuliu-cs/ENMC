import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dre
import numpy as np
from quantize import quantize
from sitq import Mips
import hnswlib
import time


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        # for DRE
        self.enable_dre = False
        self.dre_inference = False
        self.qtize = None
        self.num_candidates = None
        self.profiling = False
        self.method = 'dre'

    def dre_init(self, scale=0.25, device='cpu', method='dre', num_clusters=None):
        w = self.decoder.weight
        if method == 'dre':
            self.register_buffer('proj_matrix', dre.fit(
                w.shape[0], w.shape[1], scale=scale, dist='ternary').to(device))
            self.register_parameter('screen_w', nn.Parameter(
                dre.srp(w.data, getattr(self, 'proj_matrix')) ))
            self.register_parameter('screen_b', nn.Parameter(
                self.decoder.bias.data ))
        elif method == 'svd':
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            print('SVD: ', u.shape, s.shape, vh.shape)
            # print('Original: ', w)
            # print('Recovered: ', torch.matmul(u @ torch.diag(s), vh))
            self.register_buffer('preview_w', u @ torch.diag(s))
            self.register_buffer('vh', vh)
            self.register_buffer('b', self.decoder.bias.data)        
        elif method == 'knn':
            print('Using KNN search')
            n, d = w.shape[0], w.shape[1]
            W = w.data.cpu().numpy()
            G = hnswlib.Index(space='ip', dim=d)
            G.init_index(max_elements=n, ef_construction=200, M=16)
            indices = np.arange(n)
            G.add_items(W, indices)
            G.set_ef(50)
            self.graph = G
            print('Graph construction done')

            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            print('SVD: ', u.shape, s.shape, vh.shape)
            self.register_buffer('preview_w', u @ torch.diag(s))
            self.register_buffer('vh', vh)
            self.register_buffer('b', self.decoder.bias.data)
        elif method == 'mips':
            print('Using MIPS search')
            n, d = w.shape[0], w.shape[1]
            W = w.data.cpu().numpy()
            self.mips = Mips(signature_size=8)
            self.mips.fit(W)
            print('MIPS construction done')

            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            print('SVD: ', u.shape, s.shape, vh.shape)
            self.register_buffer('preview_w', u @ torch.diag(s))
            self.register_buffer('vh', vh)
            self.register_buffer('b', self.decoder.bias.data)

    def dre_activate(self, enable_dre=False, threshold=None, dre_inference=False,
            qtize=None, num_candidates=None, preview=None, profiling=False, method='dre'):
        self.num_candidates = num_candidates
        if method == 'dre':
            self.enable_dre = enable_dre
            self.profiling = profiling
            self.dre_inference = dre_inference
            screen_w = getattr(self, 'screen_w')
            screen_w.requires_grad = enable_dre
            screen_b = getattr(self, 'screen_b')
            screen_b.requires_grad = enable_dre
            if qtize:
                print('Quantize sreen W to INT-{}'.format(qtize))
                screen_w.data = quantize(screen_w.data, num_bits=qtize, dequantize=True, signed=True)
                self.qtize = qtize
        elif method == 'svd':
            self.enable_dre = enable_dre
            self.profiling = profiling
            self.preview = preview
            self.method = 'svd'
            if qtize:
                print('Quantize SVD u, s, vh to INT-{}'.format(qtize))
                pw = getattr(self, 'preview_w')
                vh = getattr(self, 'vh')
                pw.data = quantize(pw.data, num_bits=qtize, dequantize=True, signed=True)
                vh.data = quantize(vh.data, num_bits=qtize, dequantize=True, signed=True)
        elif method == 'knn':
            self.enable_dre = enable_dre
            self.profiling = profiling
            self.method = 'knn'
            self.preview = preview
        elif method == 'mips':
            self.enable_dre = enable_dre
            self.profiling = profiling
            self.method = 'mips'
            self.preview = preview

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        # t_start = time.time()
        emb = self.drop(self.encoder(input))
        # print('encoder: ', time.time() - t_start)
        # t_start = time.time()
        output, hidden = self.rnn(emb, hidden)
        # print('rnn: ', time.time() - t_start)
        output = self.drop(output)

        if self.enable_dre:
            x = output.clone().detach()
            if self.method == 'dre':
                if self.qtize:
                    if self.profiling: np.save('input_fp32', x.cpu().numpy())
                    x = quantize(x, num_bits=self.qtize, dequantize=True, signed=True)
                    if self.profiling: np.save('input_fixed4', x.cpu().numpy())
                x = dre.srp(x, getattr(self, 'proj_matrix'))
                z = F.linear(x, getattr(self, 'screen_w'), getattr(self, 'screen_b'))

                decoded = self.decoder(output)
                t = decoded
                decoded = decoded.view(-1, self.ntoken)
                
                if self.dre_inference:
                    z = z.view(-1, self.ntoken)
                    topk_v, _ = torch.topk(z, self.num_candidates, dim=-1)
                    topk_th = topk_v[:,-1].unsqueeze(-1).expand_as(z)
                    m = torch.gt(z, topk_th).to(torch.float)
                    if self.profiling:
                        np.save('candidates_mask', m.cpu().numpy())
                        np.save('candidates_threshold', topk_th.cpu().numpy())
                        np.save('classifier_weight', self.decoder.weight.data.cpu().numpy())
                        np.save('classifier_bias', self.decoder.bias.data.cpu().numpy())
                        np.save('screener_weight', getattr(self, 'screen_w').data.cpu().numpy())
                        np.save('screener_bias', getattr(self, 'screen_b').data.cpu().numpy())
                        self.profiling = False
                    decoded = torch.mul(decoded, m) + torch.mul(z, 1.0 - m) 

                return F.log_softmax(decoded, dim=1), hidden, (t, z)
            elif self.method == 'svd':
                if self.qtize:
                    x = quantize(x, num_bits=self.qtize, dequantize=True, signed=True)
                x = F.linear(x, getattr(self, 'vh'))
                pw = getattr(self, 'preview_w')
                z = F.linear(x[:,:,:self.preview], pw[:,:self.preview], getattr(self, 'b'))
                topk_v, _ = torch.topk(z, self.num_candidates, dim=-1)
                topk_th = topk_v[:,:,-1].unsqueeze(-1).expand_as(z)
                m = torch.gt(z, topk_th).to(torch.float)
                z_full = F.linear(x, pw, getattr(self, 'b'))
                z = torch.mul(z_full, m) + torch.mul(z, 1.0 - m)
                if self.profiling:
                    print(x.shape, pw.shape)
                    print(z.shape)
                    self.profiling = False
                return F.log_softmax(z.view(-1, self.ntoken), dim=1), hidden, (None, None)
            elif self.method == 'knn':
                x_shape = x.size()
                h = torch.reshape(x, [-1, x_shape[-1]])
                decoded = self.decoder(output)
                decoded = decoded.view(-1, self.ntoken)

                # KNN search for indices
                knn_indices, _ = self.graph.knn_query(h.cpu().numpy(), k=self.num_candidates)
                m = torch.zeros(decoded.shape)
                m.scatter_(-1, torch.from_numpy(knn_indices.astype(np.int64)), 1.)
                m = m.to(torch.device('cuda:0'))

                # Using SVD approximation for unselected
                x = F.linear(x, getattr(self, 'vh'))
                pw = getattr(self, 'preview_w')
                z = F.linear(x[:,:,:self.preview], pw[:,:self.preview], getattr(self, 'b'))

                if self.profiling:
                    print(x.shape, knn_indices.shape)
                    np.save('knn_mask', m.cpu().numpy())
                    self.profiling = False
                decoded = torch.mul(decoded, m) + torch.mul(z.view(-1, self.ntoken), 1. - m)
                return F.log_softmax(decoded, dim=1), hidden, (None, None)
            elif self.method == 'mips':
                x_shape = x.size()
                h = torch.reshape(x, [-1, x_shape[-1]])
                decoded = self.decoder(output)
                decoded = decoded.view(-1, self.ntoken)

                # MIPS search for indices
                Q = h.cpu().numpy()
                M = list()
                for q in Q:
                    I, _ = self.mips.search(q, limit=self.num_candidates, require_items=True)
                    m = torch.zeros(self.ntoken)
                    m.scatter_(-1, torch.from_numpy(I), 1.)
                    M.append(m)
                M = torch.stack(M)
                M = M.to(torch.device('cuda:0'))

                # Using SVD approximation for unselected
                x = F.linear(x, getattr(self, 'vh'))
                pw = getattr(self, 'preview_w')
                z = F.linear(x[:,:,:self.preview], pw[:,:self.preview], getattr(self, 'b'))

                if self.profiling:
                    print(x.shape, M.shape)
                    print('Mask: {:.5f}'.format(M.sum() / M.numel()))
                    # np.save('mips_mask', m.cpu().numpy())
                    self.profiling = False
                decoded = torch.mul(decoded, M) + torch.mul(z.view(-1, self.ntoken), 1. - M)
                return F.log_softmax(decoded, dim=1), hidden, (None, None)
        else:
            # t_start = time.time()
            decoded = self.decoder(output)
            # print('decoder: ', time.time() - t_start)
            decoded = decoded.view(-1, self.ntoken)
            return F.log_softmax(decoded, dim=1), hidden, (None, None)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
