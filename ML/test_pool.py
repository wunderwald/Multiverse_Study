from multiprocessing import Pool

def f(args):
    print(f"{args['a']} - {args['b']} - {args['c']}, index={args['index']}")

if __name__ == '__main__':
    N = 10
    fixed_params = {'a': 222, 'b': 'xxx', 'c': 777}
    with Pool() as pool:
        params = [{**fixed_params, 'index': i} for i in range(N)]
        pool.map(f, params)