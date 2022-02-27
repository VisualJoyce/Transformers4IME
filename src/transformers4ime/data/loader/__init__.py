"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

A prefetch loader to speedup data loading
Modified from Nvidia Deep Learning Examples
(https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch).
"""
import glob
import importlib
import os
from abc import abstractmethod

import lmdb
import msgpack
import torch
from lz4.frame import decompress
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers4ime.utils.const import BUCKET_SIZE
from transformers4ime.utils.logger import LOGGER


def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


class PinyinGPT2Loader(object):

    def __init__(self, opts):
        self.dataset_cls = DATA_REGISTRY[opts.dataset_cls]
        self.eval_dataset_cls = DATA_REGISTRY[opts.eval_dataset_cls]
        self.config = opts

    @abstractmethod
    def context_parsing(self, key, len_list):
        raise NotImplementedError

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        # overall_start = dataset.start
        # overall_end = dataset.end
        # configure the dataset to only process the split workload
        # per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        # worker_id = worker_info.id
        # dataset.start = overall_start + worker_id * per_worker
        # dataset.end = min(dataset.start + per_worker, overall_end)
        dataset.ids = dataset.ids[worker_info.id::worker_info.num_workers]
        dataset.lens = dataset.lens[worker_info.id::worker_info.num_workers]

    def create_dataloader(self, split, db_dir, lens, ids):
        dataset_cls = self.dataset_cls if split == 'train' else self.eval_dataset_cls
        # batch_size = getattr(opts, f'{split}_batch_size') if split == 'train' else opts.val_batch_size
        batch_size = self.config.train_batch_size if split == 'train' else self.config.val_batch_size
        # create_dataloader_fn = create_dataloader_fn_dict.get(split, create_dataloader)
        LOGGER.info(f"Loading {split} Dataset using {dataset_cls}")
        opts = self.config
        dset = dataset_cls(split, db_dir, lens, ids, opts)
        # sampler = DistributedTokenBucketSampler(
        #     opts.size, opts.rank, dset.lens,
        #     bucket_size=BUCKET_SIZE,
        #     batch_size=batch_size,
        #     droplast='train' == split,
        #     size_multiple=opts.size_multiple)
        sampler = TokenBucketSampler(
            dset.lens,
            bucket_size=BUCKET_SIZE,
            batch_size=batch_size,
            droplast='train' == split,
            size_multiple=opts.size_multiple)
        LOGGER.info(f"Loading {split} Dataset using {opts.n_workers} workers!")
        loader = DataLoader(dset, batch_sampler=sampler,
                            num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                            # worker_init_fn=worker_init_fn,
                            collate_fn=dataset_cls.collate_fn)
        return PrefetchLoader(loader)

    def create_dataloaders(self, splits=None):
        opts = self.config
        if splits is None:
            splits = []
            for k in dir(opts):
                if k.endswith('_txt_db'):
                    splits.append(k.replace('_txt_db', ''))

        LOGGER.info(f"Dataset splits: {splits}")

        dataloaders = {}
        for split in [s for s in splits if s != 'train']:
            # txt_db = getattr(opts, f'{split}_txt_db')

            db_dir = os.path.join('/txt',
                                  intermediate_dir(opts.pretrained_model_name_or_path),
                                  opts.annotator_cls,
                                  f'benchmarks-validation-val_txt_db.0')
            LOGGER.info(f"Loading train Dataset file2lens from {db_dir}")

            env = lmdb.open(db_dir,
                            readonly=True, create=False, lock=False, max_dbs=2,
                            readahead=not _check_distributed())
            db_lens = env.open_db('lens'.encode())
            with env.begin(buffers=True) as txn:
                cursor_lens = txn.cursor(db_lens)
                lens = []
                ids = []
                for k, _lens in tqdm(cursor_lens.iternext(keys=True, values=True),
                                     total=txn.stat(db_lens)['entries'],
                                     desc="Prepare text spans"):
                    # new id and len
                    k = k.tobytes()
                    len_list = msgpack.loads(decompress(_lens), raw=False)
                    try:
                        _id, _len = self.context_parsing(k, len_list)
                        ids.append(_id)
                        lens.append(_len)
                    except:
                        LOGGER.error(f'{k}, {len_list}')

            dataloaders[split] = self.create_dataloader(split, db_dir, lens, ids)
        return splits, dataloaders

    def iter_db(self):
        opts = self.config
        for genre in opts.genre.split(':'):
            db_prefix = os.path.join('/txt',
                                     intermediate_dir(opts.pretrained_model_name_or_path),
                                     opts.annotator_cls, f'{opts.domain}-{genre}-train_txt_db')
            db_all = glob.glob(f'{db_prefix}.*')
            for db_dir in db_all:
                yield db_dir

    def create_train_dataloaders(self):
        opts = self.config
        # txt_db = getattr(opts, f'{split}_txt_db')
        # create_dataloader_fn = create_dataloader_fn_dict.get(split, create_dataloader)
        LOGGER.info(f"Loading Train Dataset using {self.dataset_cls}")
        for db_index, db_dir in enumerate(self.iter_db()):
            if db_index % opts.n_gpu == opts.rank:
                LOGGER.info(f"Loading train Dataset file2lens from {db_dir}")
                env = lmdb.open(db_dir,
                                readonly=True, create=False, lock=False, max_dbs=2,
                                readahead=not _check_distributed())
                db_lens = env.open_db('lens'.encode())
                with env.begin(buffers=True) as txn:
                    LOGGER.info(f"Open db lens with length {txn.stat(db_lens)['entries']}.")
                    cursor_lens = txn.cursor(db_lens)
                    lens = []
                    ids = []
                    for j, (k, _lens) in enumerate(cursor_lens.iternext(keys=True, values=True)):
                        k = k.tobytes()
                        len_list = msgpack.loads(decompress(_lens), raw=False)
                        try:
                            _id, _len = self.context_parsing(k, len_list)
                        except Exception as e:
                            LOGGER.warn(e, k, len_list)
                            continue

                        ids.append(_id)
                        lens.append(_len)

                        if len(ids) % opts.max_dataloader_size == 0:
                            LOGGER.info(f"Prepare text spans: {j} from {db_dir}")
                            yield self.create_dataloader('train', db_dir, lens, ids)
                            lens = []
                            ids = []

                    if lens and ids:
                        yield self.create_dataloader('train', db_dir, lens, ids)


LOADER_REGISTRY = {}


def register_loader(name):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_loader_cls(cls):
        if name in LOADER_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        LOADER_REGISTRY[name] = cls
        return cls

    return register_loader_cls


# automatically import any Python files in the models/ directory
datasets_dir = os.path.dirname(__file__)
for file in os.listdir(datasets_dir):
    path = os.path.join(datasets_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'transformers4ime.data.loader.{model_name}')
