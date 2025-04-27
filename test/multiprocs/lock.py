from multiprocessing import Process, Lock

def test(l, i):
    # NOTE - 获取锁
    l.acquire()
    try:
        print('Hello Lock: ', i)
    finally:
        # NOTE - 确保无论如何都要成功释放锁
        l.release()


if __name__ == '__main__':
    lock = Lock()
    for i in range(10):
        Process(target=test, args=(lock, i)).start()