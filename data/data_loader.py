import data.flickr25k as flickr25k

def load_data(dataset, num_query, num_train, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    if dataset == 'flickr25k':
        root = r'/home/shiyufeng/dataset/their/flickr25k'
        query_dataloader, train_dataloader, retrieval_dataloader = flickr25k.load_data(root,
                                                                                       num_query,
                                                                                       num_train,
                                                                                       batch_size,
                                                                                       num_workers
                                                                                       )
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, train_dataloader, retrieval_dataloader
