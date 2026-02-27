# used after model is completely trained, and test for results
import json
import math
import torch
import os
import argparse
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
import warnings

mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, veh_lists, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx + 1, ...], cost[idx:idx + 1, ...], veh_lists[idx:idx + 1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result], [
        veh_lists[i] if i >= 0 else None for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model, opts.obj)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    if opts.generalization:
        model_path=opts.generalization_model_path
    else:
        model_path='outputs/{}/{}_C{}_RC{}_V{}_{}/hecvrp_rollout'.format(opts.obj,opts.problem,opts.customer_size,opts.charger_size,opts.n_veh,opts.obj)

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset_path='instance/V{}/C{}_RC{}_V{}/'.format(opts.n_veh,opts.customer_size,opts.charger_size,opts.n_veh)
        results={
            'name':[],
            'cost':[],
            'time':[],
            'sequence':[],
            'veh_list':[],

        }
        with open(dataset_path+os.listdir(dataset_path)[0],'r') as f:
            data=json.load(f)
            VEHICLE_CAPACITY=data['Capacity']
            VEHICLE_BATTERY=data['Battery']
            SPEED=data['Speed']
        model, _ = load_model(model_path, opts, VEHICLE_CAPACITY, VEHICLE_BATTERY, SPEED, )
        for p in os.listdir(dataset_path):
            results['name'].append(p)
            path=dataset_path+p
            dataset = model.problem.make_dataset(VEHICLE_CAPACITY,VEHICLE_BATTERY,filename=path)
            sequence, cost, veh_list,duration = _eval_dataset(model, dataset, width, softmax_temp, opts, device)
            results['sequence'].append(sequence)
            results['cost'].append(cost[0])
            results['veh_list'].append(veh_list)
            results['time'].append(duration)
        if opts.generalization:
            save_path = './results/V{}/Gmodel-{}_C{}_RC{}_V{}/'.format(opts.n_veh,opts.model_name, opts.customer_size, opts.charger_size,
                                                             opts.n_veh)
        else:
            save_path='./results/V{}/C{}_RC{}_V{}/'.format(opts.n_veh,opts.customer_size,opts.charger_size,opts.n_veh)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df=pd.DataFrame(results)
        df.to_csv(save_path+f'{opts.obj}_{opts.decode_strategy}{opts.width}.csv')



def _eval_dataset(model, dataset, width, softmax_temp, opts, device):
    # print('data', dataset[0])
    model.to(device)
    model.eval()
    VEHICLE_CAPACITY=model.VEHICLE_CAPACITY
    SPEED=model.SPEED
    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=1)

    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)
        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    # assert width == 0, "Do not set width when using greedy"
                    # assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                    #     "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs, veh_lists = model.sample_many(batch, VEHICLE_CAPACITY,SPEED,batch_rep=batch_rep, iter_rep=iter_rep)
                print('cost', costs)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
            sequences, costs, veh_lists = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(), veh_lists.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        else:
            sequences, costs, veh_lists = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(), veh_lists.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )

        duration = time.time() - start

    return sequences[0], costs, veh_lists[0],duration


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='hecvrp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--charger_size', type=int, default=3, help="The size of the charger")
    parser.add_argument('--customer_size', type=int, default=20, help="The size of the customer")
    parser.add_argument('--obj', default='min-sum')
    parser.add_argument('--n_veh',default=3,help='number of vehicle')
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--width',default='10000', type=int,
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--generalization',action='store_true')
    parser.add_argument('--generalization_model_path',default='outputs/min-max/hecvrp_C10_RC3_V3_min-max/hecvrp_rollout',type=str)
    parser.add_argument('--model_name',type=str,default=10)
    parser.add_argument('--decode_strategy', default='sample',type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")

    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    widths = opts.width if opts.width is not None else [0]

    eval_dataset(opts.width, opts.softmax_temperature, opts)
