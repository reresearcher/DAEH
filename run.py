import torch
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from loguru import logger

import daeh

from data.data_loader import load_data
from model_loader import load_model


multi_labels_dataset = [
    'flickr25k',
    'COCO',
    'NUSWIDE',
]

label_dim = {
    'NUSWIDE':21,
    'flickr25k':24,
    'COCO':91
}

txt_dim = {
    'NUSWIDE':1000,
    'flickr25k':1386,
    'COCO':2000
}

feature_dim = {
    'alexnet': 4096,
    'vgg16': 4096,
    'resnet152': 2048,
    'transformer': 768,
}


def run():
    # Load configuration
    args = load_config()
    logger.add(os.path.join('logs', '{time}.log'), rotation="500 MB", level="INFO")
    logger.info(args)

    # Load dataset
    query_dataloader, train_dataloder, retrieval_dataloader = load_data(args.dataset,
                                                                        args.num_query,
                                                                        args.num_train,
                                                                        args.batch_size,
                                                                        args.num_workers,
                                                                        )

    multi_labels = args.dataset in multi_labels_dataset
    daeh.process(
        args.near_neighbor,
        args.num_train,
        args.batch_size,
        args.dataset,
        train_dataloder,
        query_dataloader,
        retrieval_dataloader,
        multi_labels,
        args.code_length,
        feature_dim[args.arch],
        label_dim[args.dataset],
        args.alpha,
        args.beta,
        args.gamma,
        args.max_iter,
        args.arch,
        args.lr,
        args.device,
        args.verbose,
        args.evaluate_interval,
        args.snapshot_interval,
        args.topk,
        txt_dim[args.dataset],
        args.ceshi_tri,
        args.ceshi_cross,
        args.a1,
        args.ceshi_tea
        )


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='SSDH_PyTorch')
    parser.add_argument('-g', '--gpu', default='0', type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('-d', '--dataset', default='nus-wide-tc-21', type=str,
                        help='Dataset name.')
    parser.add_argument('-c', '--code-length', default=16, type=int,
                        help='Binary hash code length.(default: 16)')
    parser.add_argument('-o', '--near_neighbor', default=3, type=int,
                        help='Number of neighbors.(default: 3)')
    parser.add_argument('-T', '--max-iter', default=30, type=int,
                        help='Number of iterations.(default: 30)')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        help='Learning rate.(default: 1e-3)')
                        
    parser.add_argument('-q', '--num-query', default=2000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('-t', '--num-train', default=10000, type=int,
                        help='Number of training data points.(default: 5000)')

    parser.add_argument('-w', '--num-workers', default=2, type=int,
                        help='Number of loading data threads.(default: 4)')
    parser.add_argument('-b', '--batch-size', default=18, type=int,
                        help='Batch size.(default: 32)')

    parser.add_argument('-a', '--arch', default='transformer', type=str,
                        help='CNN architecture.(default: vgg16)')
    parser.add_argument('-k', '--topk', default=5000, type=int,
                        help='Calculate map of top k.(default: 5000)')

    parser.add_argument('-v', '--verbose', default=True,
                        help='Print log.')

    parser.add_argument('--train', action='store_true',
                        help='Training mode.')

    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate mode.')
    parser.add_argument('-C', '--checkpoint', default=None, type=str,
                        help='Path of checkpoint.')


    parser.add_argument('--resume', action='store_true',
                        help='Resume mode.')


    parser.add_argument('-e', '--evaluate-interval', default=5, type=int,
                        help='Interval of evaluation.(default: 500)')
    parser.add_argument('-s', '--snapshot-interval', default=50, type=int,
                        help='Interval of evaluation.(default: 800)')
                    
    parser.add_argument('--alpha', default=2, type=float,
                        help='Hyper-parameter.(default:2)')
    parser.add_argument('--beta', default=2, type=float,
                        help='Hyper-parameter.(default:2)')
                        
    
    parser.add_argument('--gamma', default=-3, type=float,
                        help='Hyper-parameter.(default:2)')
    parser.add_argument('--ceshi-tri', default=4, type=float,
                        help='Hyper-parameter.(default:2)')
    parser.add_argument('--ceshi-tea', default=1, type=float,
    help='Hyper-parameter.(default:2)')
    parser.add_argument('--ceshi-cross', default=0.1, type=float,
                        help='Hyper-parameter.(default:2)')
    parser.add_argument('--a1', default=0.3, type=float,
                        help='Hyper-parameter.(default:2)')
    
    
    
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args


if __name__ == '__main__':
    run()
