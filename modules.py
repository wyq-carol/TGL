import torch
import dgl
from memorys import *
from layers.tgn_layer import *
import time

class GNNTimer:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        # print(f"Time for this step: {elapsed_time:.6f} seconds")

class GeneralModel_fusion(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False):
        super(GeneralModel_fusion, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'rnn':
                self.memory_updater = RNNMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], train_param)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer_fusion(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer_fusion(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
        elif gnn_param['arch'] == 'identity':
            self.gnn_param['layer'] = 1
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = IdentityNormLayer(self.dim_node_input)
                if 'time_transform' in gnn_param and gnn_param['time_transform'] == 'JODIE':
                    self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(gnn_param['dim_out'])
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])
    
    def forward(self, mfgs, col_ptr, row_ind, neg_samples=1):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0]) # GRUMemeoryUpdater
        out = list()
        # breakpoint()
        """
        # rst = self.layers['l0h0'](mfgs[0][0]) # /home/TGL/config/TGN.yml
        TransfomerAttentionLayer(
            (dropout): Dropout(p=0.2, inplace=False)
            (att_dropout): Dropout(p=0.2, inplace=False)
            (att_act): LeakyReLU(negative_slope=0.2)
            (time_enc): TimeEncode(
                (w): Linear(in_features=1, out_features=100, bias=True)
            )
            (w_q): Linear(in_features=200, out_features=100, bias=True)
            (w_k): Linear(in_features=372, out_features=100, bias=True)
            (w_v): Linear(in_features=372, out_features=100, bias=True)
            (w_out): Linear(in_features=200, out_features=100, bias=True)
            (layer_norm): LayerNorm((100,), eps=1e-05, elementwise_affine=True)
        )
        Block(num_src_nodes=1203, num_dst_nodes=600, num_edges=603)
        """
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](
                    b=mfgs[l][h],
                    col_ptr=col_ptr, 
                    row_ind=row_ind)
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        ret = self.edge_predictor(out, neg_samples=neg_samples)
        return ret

    def get_emb(self, mfgs):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out

class GeneralModel(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False):
        super(GeneralModel, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'rnn':
                self.memory_updater = RNNMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], train_param)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
        elif gnn_param['arch'] == 'identity':
            self.gnn_param['layer'] = 1
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = IdentityNormLayer(self.dim_node_input)
                if 'time_transform' in gnn_param and gnn_param['time_transform'] == 'JODIE':
                    self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(gnn_param['dim_out'])
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])
    
    def forward(self, mfgs, neg_samples=1):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0]) # GRUMemeoryUpdater
        out = list()
        # breakpoint()
        """
        # rst = self.layers['l0h0'](mfgs[0][0]) # /home/TGL/config/TGN.yml
        TransfomerAttentionLayer(
            (dropout): Dropout(p=0.2, inplace=False)
            (att_dropout): Dropout(p=0.2, inplace=False)
            (att_act): LeakyReLU(negative_slope=0.2)
            (time_enc): TimeEncode(
                (w): Linear(in_features=1, out_features=100, bias=True)
            )
            (w_q): Linear(in_features=200, out_features=100, bias=True)
            (w_k): Linear(in_features=372, out_features=100, bias=True)
            (w_v): Linear(in_features=372, out_features=100, bias=True)
            (w_out): Linear(in_features=200, out_features=100, bias=True)
            (layer_norm): LayerNorm((100,), eps=1e-05, elementwise_affine=True)
        )
        Block(num_src_nodes=1203, num_dst_nodes=600, num_edges=603)
        """
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        ret = self.edge_predictor(out, neg_samples=neg_samples)
        return ret

    def get_emb(self, mfgs):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out

class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x