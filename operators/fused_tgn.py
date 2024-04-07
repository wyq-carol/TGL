from torch.utils.cpp_extension import load
import torch
import os

# path = os.path.join(os.path.dirname(__file__))
path=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
path=os.path.join(path,'src')

fused_tgn_SM = load(
    name="fused_tgn_SM",
    sources=[os.path.join(path, "fused_tgn/fused_tgn_SM.cpp"), os.path.join(path, "fused_tgn/fused_tgn_SM.cu")],
    verbose=False,
)

# TODO@mkj
def fused_tgn_op(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat,save_memory=True):
    return FusedTGNFunction_SM.apply(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat)

# TODO@mkj
class FusedTGNFunction_SM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat):
        out_feat,edge_max,edge_sum=fused_tgn_SM.tgn_forward(attn_row,attn_col,row_ptr,col_ind,negative_slope,in_feat)
        ctx.save_for_backward(row_ptr,col_ind,col_ptr,row_ind,edge_max,edge_sum,in_feat,attn_row,attn_col)
        ctx.negative_slope=negative_slope      
        return out_feat
    
    @staticmethod
    def backward(ctx,grad_out):
        row_ptr,col_ind,col_ptr,row_ind,edge_max,edge_sum,in_feat,attn_row,attn_col=ctx.saved_tensors
        grad_out=grad_out.contiguous()
        # print('start backward')
        grad_feat,grad_attn_row,grad_attn_col=fused_tgn_SM.tgn_backward(
            ctx.negative_slope,row_ptr,col_ind,col_ptr,row_ind,edge_max,edge_sum,in_feat,attn_row,attn_col,grad_out)
        # print('end backward')
        # print(torch.isnan(grad_feat).sum())
        # print(torch.isnan(grad_attn_row).sum())
        # print(torch.isnan(grad_attn_col).sum())
        return grad_attn_row,grad_attn_col,None,None,None,None,None,grad_feat,None
