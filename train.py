import argparse
import os
from torch.profiler import profile, record_function, ProfilerActivity
import multiprocessing

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
from modules import *
from sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score

class GNNTimer2:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        # print(f"Time for this step: {elapsed_time:.6f} seconds")

class GNNTimer:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Time for this step: {elapsed_time:.6f} seconds")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# set_seed(0)

node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
g, df = load_graph(args.data)
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]
    
    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    train_node_set = set(np.unique(np.hstack([train_df['src'].values, train_df['dst'].values])))
    new_node_set = total_node_set - train_node_set
    
    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end+index)
    
    print('Inductive links', len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds

if args.use_inductive:
    inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
    df = df.iloc[inductive_inds]
    
gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

# use_inductive?
if args.use_inductive:
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.src.values)
    print("inductive nodes", len(inductive_nodes))
    neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

def eval(mode='val'):
    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            total_loss += criterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
        if mode == 'val':
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

if not os.path.isdir('models'):
    os.mkdir('models')
if args.model_name == '':
    path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
else:
    path_saver = 'models/{}.pkl'.format(args.model_name)
best_ap = 0
best_e = 0
val_losses = list()
group_indexes = list()
group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))
# reorder? 暂无 reorder=16
if 'reorder' in train_param:
    # random chunk shceduling
    reorder = train_param['reorder']
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), train_param['batch_size'] // reorder)
    group_idx = np.tile(group_idx, train_edge_end // train_param['batch_size'] + 1)[:train_edge_end]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, train_param['reorder']):
        additional_idx = np.zeros(train_param['batch_size'] // train_param['reorder'] * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])

def calculate_dglblock_ret(rows):
    root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
    ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
    if sampler is not None:
        if 'no_neg' in sample_param and sample_param['no_neg']:
            pos_root_end = root_nodes.shape[0] * 2 // 3
            sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
        else:
            sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
        # time_sample += ret[0].sample_time()
    return root_nodes, ts, ret

def prefetch_batch(grouped_data, prefetch_queue):
    for _, rows in grouped_data:
        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        if sampler is not None:
            if 'no_neg' in sample_param and sample_param['no_neg']:
                pos_root_end = root_nodes.shape[0] * 2 // 3
                sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
            else:
                sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            # time_sample += ret[0].sample_time()
            mfgs = to_dgl_blocks(ret, sample_param['history'])
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
        data_pack = (root_nodes, ts, mfgs)    
        prefetch_queue.put(data_pack)
        # calculate_dglblock_ret(rows)



# training 
"""
for e in range(train_param['epoch']):
    print('Epoch {:d}:'.format(e))
    # with GNNTimer2():
    time_sample = 0
    time_prep = 0
    time_tot = 0
    total_loss = 0
    # training
    model.train()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
    for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
        with GNNTimer2():
            # train_epoch()
            with GNNTimer():
            # sample 得到 DGLBLOCK
                t_tot_s = time.time()
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
                ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = root_nodes.shape[0] * 2 // 3
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()
                    time_sample += ret[0].sample_time()
                t_prep_s = time.time()
            with GNNTimer():
            # prep mfgs
                if gnn_param['arch'] != 'identity':
                    mfgs = to_dgl_blocks(ret, sample_param['history'])
                else:
                    mfgs = node_to_dgl_blocks(root_nodes, ts)
                mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0])
                time_prep += time.time() - t_prep_s
            with GNNTimer():
                optimizer.zero_grad()
            with GNNTimer():
            # 前向
                pred_pos, pred_neg = model(mfgs)
            with GNNTimer():
            # 计算loss
                loss = criterion(pred_pos, torch.ones_like(pred_pos))
                loss += criterion(pred_neg, torch.zeros_like(pred_neg))
                total_loss += float(loss) * train_param['batch_size']
            with GNNTimer():
            # 反向 优化不了一点
                loss.backward()
                optimizer.step()
            t_prep_s = time.time()
            with GNNTimer():
                if mailbox is not None:
                    eid = rows['Unnamed: 0'].values
                    mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
            time_prep += time.time() - t_prep_s # memory module相关的时间
            time_tot += time.time() - t_tot_s # 每轮epoch的时间
            print()
            print()
"""
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

if __name__ == '__main__':
    # TGN 简化版
    # 主程序逻辑
    if not multiprocessing.get_start_method(allow_none=True):
        multiprocessing.set_start_method("spawn")

    for e in range(train_param['epoch']):
        print('Epoch {:d}:'.format(e))
        t_tot_s = 0
        t_prep_s = 0
        time_sample = 0
        time_prep = 0
        
        time_tot = 0
        total_loss = 0
        
        # training
        model.train()
        if sampler is not None:
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
            model.memory_updater.last_updated_nid = None
            
        random_index = random.randint(0, len(group_indexes) - 1)
        grouped_data = df[:train_edge_end].groupby(group_indexes[random_index])
        prefetch_queue = multiprocessing.Queue(maxsize=1)
        
        p_prefetch = multiprocessing.Process(target=prefetch_batch, args=(grouped_data, prefetch_queue))
        p_prefetch.start()
        
        for idx, rows in grouped_data:
            t_tot_s = time.time()
            # sample 得到 DGLBLOCK -> 当前轮次在cpu上计算下一轮次然后提前swap过来
            with record_function("sample DGLBlock"):
                with GNNTimer():
                    t_get1 = time.time()
                    data_pack = prefetch_queue.get()
                    t_get2 = time.time()
                    root_nodes, ts, mfgs = data_pack
                    
                    data_pack = (root_nodes, ts, mfgs) 
                    # import pickle
                    # shared_mfg = pickle.dumps(data_pack)
                    # print(shared_mfg)
                    # TODO 解决无法流式的问题
                    # root_nodes, ts, ret = calculate_dglblock_ret(rows)
                    t_prep_s = time.time()
            # prep mfgs
            with record_function("prepare mfgs"):
                with GNNTimer():
                    print(f"prefetch data: {mfgs}")
                    print(f"prefetch time: {t_get2 - t_get1}")
                    # breakpoint()
                    mailbox.prep_input_mails(mfgs[0])
                    time_prep += time.time() - t_prep_s
            with record_function("train forward"):
                with GNNTimer():
                    optimizer.zero_grad()
                with GNNTimer():
                    # 前向
                    pred_pos, pred_neg = model(mfgs)
            with record_function("train loss"):
                # 计算loss
                with GNNTimer():
                    loss = criterion(pred_pos, torch.ones_like(pred_pos))
                    loss += criterion(pred_neg, torch.zeros_like(pred_neg))
                    total_loss += float(loss) * train_param['batch_size']
            with record_function("train backward"):
                with GNNTimer():
                    # 反向 优化不了一点
                    loss.backward()
                    optimizer.step()
                t_prep_s = time.time()
            with record_function("update memory"):
                with GNNTimer():
                    eid = rows['Unnamed: 0'].values
                    mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                    block = None
                    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
                time_prep += time.time() - t_prep_s # memory module相关的时间
                time_tot += time.time() - t_tot_s # 每轮epoch的时间
            print()
            print()
        p_prefetch.join()
        ap, auc = eval('val')
        if e > 2 and ap > best_ap:
            best_e = e
            best_ap = ap
            torch.save(model.state_dict(), path_saver)
        print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
        print('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s'.format(time_tot, time_sample, time_prep))


    print('Loading model at epoch {}...'.format(best_e))
    model.load_state_dict(torch.load(path_saver))
    model.eval()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
        eval('train')
        eval('val')
    ap, auc = eval('test')
    if args.eval_neg_samples > 1:
        print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
    else:
        print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
