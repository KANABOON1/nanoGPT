"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and save the data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B"
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)   # 100M per shards

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset, 默认下载到 ~/.cache/huggingface/datasets/fineweb-edu/sample-10BT/ 中
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # endoftext token, serves as a doc spliter
def tokenize(doc) -> np.ndarray:
    # tokenize a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all()  # 确保所有 token 都在可以表示的范围内
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    # # writes a numpy array of uint16 tokens to a binary file
    # with open(filename, 'wb') as f:
    #     f.write(tokens_np.tobytes())
    np.save(filename, tokens_np)

# tokenize all documents and write output shards
nprocs = max(1, os.cpu_count() // 2)

# NOTE - 设置进程池上下文
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # NOTE - skill: preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar: tqdm = None

    # NOTE - Iterable map, 在调用时所有子进程已经开始了运算, 每次将 chunksize 个 fw 中的元素分配给某个进程
    # iterable 的性质体现在取结果上, 而不是在计算顺序上.
    # NOTE - 虽然取结果采用的是 iterable 的形式, 但是取出结果的顺序仍然是按照输入的顺序的
    # 与 map 的区别是: 
    # map 必须等到所有结果 (e.g. 1, 2, 3...) 都计算完毕才可以返回
    # imap 只要在前面的进程计算完毕就可以取结果了 (e.g. 1 完成, 2, 3没完成, 可以返回 1 中的结果)
    # imap 计算流程: 子进程领取 chunksize 个任务, 计算完毕后将结果返回给主进程 (子进程不保留结果), 再领取 chunksize 个任务
    # chunksize: 子进程领取的 task 数, 但是返回时是按照单个任务进行返回
    for tokens in pool.imap(tokenize, fw, chunksize=16):  

        if token_count + len(tokens) < shard_size:
            # append to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')
            progress_bar.update(len(tokens))

        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")

            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)

            # update 
            shard_index += 1
            progress_bar = None  # NOTE - 会导致 progress bar 的进度条更新不满, 因为 remainder 部分并没有更新进去
            all_tokens_np[:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # NOTE - write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])