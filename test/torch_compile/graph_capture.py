"""
本模块研究计算图捕获: fx, jit, dynamo
参考自知乎 游凯超 文章: https://zhuanlan.zhihu.com/p/644590863

fx: 符号追踪, 捕获计算图, 弱, 仅能追踪输入数据, 无法追踪输入数据中的细节, 仅能追踪输入数据中的形状. 
    缺陷: 1. fx 的视角来看, 输入的 x 就是一个不知道具体参数的 tensor, 凡是涉及到输入的具体参数的步骤都会报错
          2. fx 无法除了除了 x 以外的代码
jit: 即时追踪
dynamo: 动态优化追踪(守卫条件和字节码修改)
"""
import torch

def f(x):
    return (x.relu() + 1) * x.pow(5)

def conditional_computation(x):
    """graph capturing 难点: 条件结构的函数
    """
    # NOTE - python style code, hard and inefficient for graph capturing
    # if x.sum() < 0:
    #     return x + 1
    # else:
    #     return x - 1

    # NOTE - torch style code, efficient for graph capturing
    flag = torch.lt(x.sum(), 0).to(dtype=x.dtype)   # tensor in, tensor out => 可以被fx追踪
    return flag * (x + 1) + (1 - flag) * (x - 1)


def shape_dependent(x):
    """graph capturing 难点: 形状相关的函数
    """
    # NOTE - python style code, hard and inefficient for graph capturing
    # fx 不是实时捕获, 没有办法得到 x.shape
    # bsz, *sizes = x.shape
    # return x.reshape(bsz, -1)
    
    # NOTE - torch style code, efficient for graph capturing, 不依赖于 tensor 的具体参数
    return x.flatten(start_dim=1)

def external_code(x):
    """graph capturing 难点: 外部代码的函数
    """
    # NOTE - python style code, hard and inefficient for graph capturing
    # fx 无法处理与 x 无关的代码
    # import numpy as np
    # return x + torch.from_numpy(np.random.random((5, 5, 5)))   # fx 只能捕获到 x 与一个 tensor 常量相加, 并不知道该 tensor 的来源(来自 random)

    # NOTE - torch style code, efficient for graph capturing, 不依赖于 tensor 的具体参数
    return x + torch.randn(5, 5, 5)

def custom_backend(gm, example_inputs):
    print(gm.compile_subgraph_reason)
    print(gm.graph)
    print(gm.code)
    return gm.forward

if __name__ == "__main__":
    # NOTE - 符号追踪, 计算图捕获的最弱办法, 在捕获期间不进行任何的运算. 只能将输入当做某个 tensor 进行追踪, 无法知道该 tensor 具体的细节 ...
    if True:
        print("----------------------- fx graph capturing (symbolic_trace) -----------------------")
        print(torch.fx.symbolic_trace(conditional_computation).code)
        print(torch.fx.symbolic_trace(shape_dependent).code)
        print(torch.fx.symbolic_trace(external_code).code)
    
    # NOTE - 及时追踪, 一般在运行第一个真实输入的数据时进行追踪, 并通过追踪一些预定义的算子来实现计算图的捕获
    # 及时追踪对输入的数据过度特化, 生成的计算图只用于输入的数据. 一旦后续的数据与捕获计算图的数据不一样, 就有可能出错 
    # e.g. 捕获计算图的数据走条件判断的A分支, 当前的输入数据走条件判断的B分支
    # NOTE - torch.jit.trace 的设计初衷是为了在训练结束之后将模型导出用于后续的推理, 对训练代码的动态性支持有限 (e.g. BN, dropout)
    if False:
        input = torch.randn(5, 5, 5)
        f_traced = torch.jit.trace(f, input)
        print(f_traced.code)
        print(f_traced.graph)
    
    # NOTE - Dynamo (动态优化)
    if True:
        print("------------------------- dynamo graph capturing (dynamic optimization)-----------------------")

        # CASE1: 条件示例: 遇到条件语句时, Dynamo 将计算图切割为多个子计算图 (包括条件语句之前的计算图, 要执行的条件分支的计算图)
        print("CASE1: condition function >>>")
        input = torch.randn(5, 5, 5)
        opt_conditional_computation = torch.compile(conditional_computation, backend=custom_backend)
        output = opt_conditional_computation(input)

        input_1 = torch.ones(5, 5, 5)
        output = opt_conditional_computation(input_1)

        input_1 = torch.ones(5, 5, 5) * -1
        output = opt_conditional_computation(input_1)
        
        # CASE2: 形状依赖示例: 由于 fx 是静态捕获, 因此无法处理 tensor 的具体参数. Dynamo 通过动态的方式可以解决这个问题
        print("CASE2: shape dependent function >>>")
        input = torch.randn(5, 5, 5)
        shape_dependent = torch.compile(shape_dependent, backend=custom_backend)
        output = shape_dependent(input)

        # CASE3: 外部代码依赖示例: fx 是静态捕获, 无法处理与 x 无关的代码, Dynamo 通过动态的方式可以解决这个问题
        print("CASE3: external code function >>>")
        input = torch.randn(5, 5, 5)
        external_code = torch.compile(external_code, backend=custom_backend)
        output = external_code(input)




