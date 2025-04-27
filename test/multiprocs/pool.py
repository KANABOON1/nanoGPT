import multiprocessing as mp
from multiprocessing import Pool
import os
import time

def f(i):
    return i**2

if __name__ == '__main__':
    # NOTE - 创建进程池: 一共有4个子进程 (并发并且并行)
    pool = Pool(processes=4)
    
    # NOTE - 将 range(10) 分配给创建的 4 个子进程
    # 每一个子进程都求值之后再汇总, 保留原输入顺序输出, 因此他必须得等到所有进程全部执行完毕之后才能返回结果
    # 阻塞主进程
    print(pool.map(f, range(10)))

    # NOTE - 不按顺序返回结果
    # 不需要等到所有进程全部执行完毕, 有结果则可以取结果
    # 用于有的进程快有的进程慢的情况
    # 阻塞主进程
    # iterable map: 当调用时所有子进程已经开始运算, iterable 的性质体现在取结果上, 哪一个结果先好就取哪个
    for i in pool.imap_unordered(f, range(10)):
        print(i)
    
    # NOTE - 将任务交给某一个子进程跑, 不会阻塞主进程
    res = pool.apply_async(f, (20,))
    print(res.get(timeout=1))   # 从子进程中取出数据, 等待1s没有数据则报错
    
    # NOTE - 异步地进行多次求值, 执行的进程可能不一样
    multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
    print([res.get() for res in multiple_results])

    
    # 让一个工作进程休眠 10 秒
    res = pool.apply_async(time.sleep, (2,))
    try:
        print(res.get(timeout=1))
    except TimeoutError:
        print("We lacked patience and got a multiprocessing.TimeoutError")
        
    print("For the moment, the pool remains available for more work")