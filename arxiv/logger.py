import torch
import numpy as np

class Logger(object):
    def __init__(self, runs, args, info=None):
        self.info = info
        self.args = args
        self.results = [[] for _ in range(runs)]
        self.initial_results = [[] for _ in range(runs)]
        self.al_results = []
        self.st_results = []
        self.alst_results = []
        if args.use_AL:
            self.al_results = [[[] for _ in range(args.num_queries)] for _i in range(runs)]
            #print('asdfasd: ',len(self.al_results))
        if args.use_ST:
            self.st_results = [[[] for _ in range(args.num_queries)] for _i in range(runs)]
        if not args.stal_order=='seq':
            self.alst_results = [[[] for _ in range(args.num_queries)] for _i in range(runs)]


    def add_init_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.initial_results)
        self.initial_results[run].append(result)

    def add_run_result(self, run):
        assert run >= 0 and run < len(self.results)
        self.results[run].append([self.initial_results,self.al_results,self.st_results])

    def add_al_result(self, run, q, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.al_results)
        self.al_results[run][q].append(result)

    def add_st_result(self, run, q, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.st_results)
        self.st_results[run][q].append(result)

    def add_alst_result(self, run, q, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.alst_results)
        self.alst_results[run][q].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            print('--- INITIAL RESULTS----')
            self.print_init_results(self.initial_results)
            if self.al_results:
                print('--- AL RESULTS----')
                self.print_results(self.al_results)
            if self.st_results:
                print('--- ST RESULTS----')
                self.print_results(self.st_results)
            if self.alst_results:
                print('--- JOINT STAL RESULTS----')
                self.print_results(self.alst_results)


    def print_results(self, results):
        result = 100 * torch.tensor(results)
        #print(result)
        best_results = []
        for r in result:
            train1 = r[:, :, 0].max().item()
            #valid = r[:,0, 1].max().item()
            valid = r[:,:, 1].max().item()
            valid_idx = np.unravel_index(r[:,:, 1].argmax(), r[:,:, 1].shape)
            train2 = r[valid_idx][0].item()
            test = r[valid_idx][2].item()
            best_results.append((train1, valid, train2, test))

        best_result = torch.tensor(best_results)
        #print('best_result', best_result)
        print(f'All runs:')
        r = best_result[:, 0]
        print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
        r = best_result[:, 1]
        print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
        r = best_result[:, 2]
        print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
        r = best_result[:, 3]
        print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

    def print_init_results(self, results):
        result = 100 * torch.tensor(results)

        #print(results)
        best_results = []
        for r in result:
            train1 = r[:, 0].max().item()
            valid = r[:, 1].max().item()
            train2 = r[r[:, 1].argmax(), 0].item()
            print('train1', train1)
            print('train2', train2)
            test = r[r[:, 1].argmax(), 2].item()
            print('test', test)
            best_results.append((train1, valid, train2, test))

        best_result = torch.tensor(best_results)
        #print(best_result)

        print(f'All runs:')
        r = best_result[:, 0]
        print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
        r = best_result[:, 1]
        print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
        r = best_result[:, 2]
        print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
        r = best_result[:, 3]
        print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
