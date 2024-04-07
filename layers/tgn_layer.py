import torch
from torch import nn
import dgl
import math
import numpy as np
import time
from tgl import sparse

from operators.fused_tgn import fused_tgn_op

"""
# test fused_tgn_op
class GATConv(nn.Module): # our gat layer
    def __init__(self,
                in_feats,
                out_feats,
                num_heads,
                negative_slope=0.2,

                ):
        super(GATConv,self).__init__()
        self.in_feats=in_feats
        self.out_feats=out_feats
        self.num_heads=num_heads
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * num_heads))
        self.attn_l = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        self.negative_slope=negative_slope
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        

    def forward(self,row_ptr,col_ind,col_ptr,row_ind,feat,save_memory=True):

        h=torch.matmul(feat,self.W).view(-1,self.num_heads,self.out_feats)

        attn_row = (self.attn_l * h).sum(dim=-1)
        attn_col = (self.attn_r * h).sum(dim=-1)
        
        out=fused_tgn_op(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,self.negative_slope,h,save_memory)
            
        return out
"""

class GNNTimer:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        # print(f"QKV: {elapsed_time:.6f} seconds")
        
class GNNTimer2:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        # print(f"softmax: {elapsed_time:.6f} seconds")

class  TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        # print("*****")
        # print(t.shape)
        # print(t.reshape((-1, 1)).shape)
        # print("*****")
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output

class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


# our
class TransfomerAttentionLayer_fusion(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False):
        super(TransfomerAttentionLayer_fusion, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.combined = combined
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if combined:
            if dim_node_feat > 0:
                self.w_q_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:
                self.w_k_e = torch.nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, node_feats, edge_feats, col_ptr_count_0, col_ptr, row_ind, num_nodes, num_edges, b):
        time_feat = self.time_enc(b.edata['dt'])
        zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
        with GNNTimer():
            # print("!!! use wyq-op")
            # breakpoint()
            
            # Block(num_src_nodes=1197, num_dst_nodes=600, num_edges=597)
            # apply vertex  b.srcdata['h']  [1197, 100]
            # apply edge    QKV             [597, 2, 50] 
            # TODO 先分别乘完再concat(不需要都乘wk 需要提前选一下按节点还是按边？)
            Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
            K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
        with GNNTimer():
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1)) 
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
        
        # TODO@mkj
        # args: node_feats, edge_feats, row_ptr, col_ind, num_nodes, num_edges, self.dim_node_feat, self.dim_edge_feat, self.dim_out
        # out=fused_tgn_op(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,self.negative_slope,h,save_memory)
        
        with GNNTimer2():
            # breakpoint()
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
            b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        # with GNNTimer():
            if self.dim_node_feat != 0:
                rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
            else:
                rst = b.dstdata['h']
            rst = self.w_out(rst)
            rst = torch.nn.functional.relu(self.dropout(rst))
        # print()
        return self.layer_norm(rst)

class TransfomerAttentionLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False):
        super(TransfomerAttentionLayer, self).__init__()
        self.wyq_test5 = True # TODO
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.combined = combined
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if combined:
            if dim_node_feat > 0:
                self.w_q_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:
                self.w_k_e = torch.nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        if self.wyq_test5:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
            with GNNTimer():
                # print("!!! use wyq-op")
                # breakpoint()
                
                # Block(num_src_nodes=1197, num_dst_nodes=600, num_edges=597)
                # apply vertex  b.srcdata['h']  [1197, 100]
                # apply edge    QKV             [597, 2, 50] 
                # TODO 先分别乘完再concat(不需要都乘wk 需要提前选一下按节点还是按边？)
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            with GNNTimer():
                Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1)) 
                K = torch.reshape(K, (K.shape[0], self.num_head, -1))
                V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            breakpoint()
            with GNNTimer2():
                # breakpoint()
                att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
                att = self.att_dropout(att)
                V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
                b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
                b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
            # with GNNTimer():
                if self.dim_node_feat != 0:
                    rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
                else:
                    rst = b.dstdata['h']
                rst = self.w_out(rst)
                rst = torch.nn.functional.relu(self.dropout(rst))
            # print()
            return self.layer_norm(rst)
        else:
            print("!!! not use wyq-op")
            # breakpoint()
            assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
            if b.num_edges() == 0:
                return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
            if self.dim_time > 0:
                # print("***** transformerlayer - edge *****")
                time_feat = self.time_enc(b.edata['dt'])
                # print("***** transformerlayer - b.num_dst *****")
                zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
            if self.combined:
                Q = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
                K = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
                V = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
                if self.dim_node_feat > 0:
                    Q += self.w_q_n(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                    K += self.w_k_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                    V += self.w_v_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                if self.dim_edge_feat > 0:
                    K += self.w_k_e(b.edata['f'])
                    V += self.w_v_e(b.edata['f'])
                if self.dim_time > 0:
                    Q += self.w_q_t(zero_time_feat)[b.edges()[1]]
                    K += self.w_k_t(time_feat)
                    V += self.w_v_t(time_feat)
                Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
                K = torch.reshape(K, (K.shape[0], self.num_head, -1))
                V = torch.reshape(V, (V.shape[0], self.num_head, -1))
                att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
                att = self.att_dropout(att)
                V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
                b.edata['v'] = V
                b.update_all(dgl.function.copy_edge('v', 'm'), dgl.function.sum('m', 'h'))
            else:
                if self.dim_time == 0 and self.dim_node_feat == 0:
                    Q = torch.ones((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
                    K = self.w_k(b.edata['f'])
                    V = self.w_v(b.edata['f'])
                elif self.dim_time == 0 and self.dim_edge_feat == 0:
                    Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                    K = self.w_k(b.srcdata['h'][b.num_dst_nodes():])
                    V = self.w_v(b.srcdata['h'][b.num_dst_nodes():])
                elif self.dim_time == 0:
                    Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                    K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                    V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                elif self.dim_node_feat == 0:
                    Q = self.w_q(zero_time_feat)[b.edges()[1]]
                    K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                    V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
                elif self.dim_edge_feat == 0:
                    Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                    K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
                    V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
                else:
                    with GNNTimer():
                        # breakpoint()
                        Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                        K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                        V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                with GNNTimer():
                    Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
                    K = torch.reshape(K, (K.shape[0], self.num_head, -1))
                    V = torch.reshape(V, (V.shape[0], self.num_head, -1))
                with GNNTimer2():
                    breakpoint()
                    att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
                    att = self.att_dropout(att)
                    V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
                    b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
                    b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
            # with GNNTimer():
                if self.dim_node_feat != 0:
                    rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
                else:
                    rst = b.dstdata['h']
                rst = self.w_out(rst)
                rst = torch.nn.functional.relu(self.dropout(rst))
            # print()
            return self.layer_norm(rst)

class IdentityNormLayer(torch.nn.Module):

    def __init__(self, dim_out):
        super(IdentityNormLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        return self.norm(b.srcdata['h'])

class JODIETimeEmbedding(torch.nn.Module):

    def __init__(self, dim_out):
        super(JODIETimeEmbedding, self).__init__()
        self.dim_out = dim_out

        class NormalLinear(torch.nn.Linear):
        # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.time_emb = NormalLinear(1, dim_out)
    
    def forward(self, h, mem_ts, ts):
        time_diff = (ts - mem_ts) / (ts + 1)
        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst
            