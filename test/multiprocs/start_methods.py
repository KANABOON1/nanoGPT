"""
学习 python 多线程/多进程 相关知识
- 多线程: 单核CPU多线程方式: 并发不并行, 在某一个时间段内, 几个程序都处于正在运行的状态, 但是具体到某一个时间点, 
只有一个程序在CPU内核上运行, 其他程序都在等待. 适用于IO型代码, 即读写文件, 网络请求, 等待时间较长的代码.
- 多进程: 多核CPU多进程方式: 并行
"""

import multiprocessing as mp
from multiprocessing import Process, Queue

def change_message(q: Queue):
    q.put('Hello')

if __name__ == '__main__':
    # NOTE - 同一个程序里只有一种启动方式
    mp.set_start_method('spawn')  
    # NOTE - 得到当前的上下文, 同一个程序中可以有多种启动方式 (e.g. fork, spawn, forkserver)
    # 一般一个程序中使用一个即可 (即使用 set_start_method)
    # ctx = mp.get_context('spawn')

    # NOTE - 主进程与子进程通讯
    q = Queue()
    p = Process(target=change_message, args=(q,))
    p.start()

    # NOTE - 默认以阻塞方式获取数据, 如果主进程获取数据时队列为空, 则会报错: Empty
    print(q.get())
 
    p.join()