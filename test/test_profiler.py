import collections
import gc
import unittest

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.testing._internal.common_utils import (
    TestCase, run_tests, TEST_WITH_ASAN, IS_WINDOWS)
from torch.autograd.profiler import profile

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(TEST_WITH_ASAN, "Cannot test with ASAN")
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilerCUDA(TestCase):
    def test_mem_leak(self):
        """Checks that there's no memory leak when using profiler with CUDA
        """
        t = torch.rand(1, 1).cuda()
        p = psutil.Process()
        last_rss = collections.deque(maxlen=5)
        for outer_idx in range(10):
            with profile(use_cuda=True):
                for _ in range(1024):
                    t = torch.mm(t, t)

            gc.collect()
            torch.cuda.empty_cache()
            last_rss.append(p.memory_info().rss)

        # with CUDA events leaking the increase in memory was ~7 MB between
        # profiler invocations above
        is_increasing = all(
            [last_rss[idx] > last_rss[idx - 1] for idx in range(1, len(last_rss))])
        max_diff = -1
        for idx in range(1, len(last_rss)):
            max_diff = max(max_diff, last_rss[idx] - last_rss[idx - 1])
        self.assertTrue(not (is_increasing and max_diff > 100 * 1024),
                        msg='memory usage is increasing, {}'.format(str(last_rss)))

class TestProfiler(TestCase):
    def test_source(self):
        """Checks that source code attribution works for eager, TS and autograd mode
        """
        # avoid automatic inlining
        prev_opt = torch._C._get_graph_executor_optimize()
        torch._C._set_graph_executor_optimize(False)

        @torch.jit.script
        def ts_method_2(x, y):
            return torch.matmul(x, y)

        @torch.jit.script
        def ts_method_1(x, y, z):
            a = x + z
            w = ts_method_2(x, y) + a
            return w.sum()

        class DummyModule(nn.Module):
            def __init__(self):
                super(DummyModule, self).__init__()
                self.conv = torch.nn.Conv2d(3, 2, kernel_size=1, stride=2, padding=3, bias=False)

            def forward(self, x):
                return self.conv(x)

        mod = DummyModule()

        with profile(with_stack=True) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            w = ts_method_1(x, y, z)
            v = 2 * w
            v.backward()
            a = torch.randn(2, 3, 2, 2, requires_grad=True)
            b = mod(a)
            c = b.sum()
            c.backward()

        print(p.key_averages(
            group_by_stack_n=5).table(
            sort_by="self_cpu_time_total", row_limit=-1))

        for e in p.function_events:
            if "aten::add" in e.name or "AddBackward" in e.name:
                self.assertTrue(any(["test_profiler" in entry for entry in e.stack]))
                self.assertTrue(any([(
                    "test_source" in entry or
                    "ts_method_1" in entry or
                    "ts_method_2" in entry) for entry in e.stack]))

        torch._C._set_graph_executor_optimize(prev_opt)

class TestProfilePython(TestCase):
    def test_python(self):
        """Checks that python side high level events are recorded.
        """
        class RepeatedDataset(torch.utils.data.Dataset):
            def __init__(self, N, D_in, D_out):
                self.N = N
                self.x = torch.randn(N, D_in)
                self.y = torch.randn(N, D_out)

            def __len__(self):
                return self.N

            def __getitem__(self, idx):
                return self.x, self.y

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super(TwoLayerNet, self).__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)

            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred

        N, D_in, H, D_out = 8, 10, 5, 2
        model = TwoLayerNet(D_in, H, D_out)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        ds = RepeatedDataset(N, D_in, D_out)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=1)
        with profile() as prof:
            for t, data in enumerate(dataloader):
                x, y = data[0], data[1]
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        cnt_dataloader_next = 0
        cnt_optimizer_step = 0
        cnt_optimizer_zerograd = 0
        for e in prof.function_events:
            if e.name == "_BaseDataLoaderIter.__next__#_BaseDataLoaderIter.__next__":
                cnt_dataloader_next += 1
            elif e.name == "SGD.step#Optimizer.step":
                cnt_optimizer_step += 1
            elif e.name == "SGD.zero_grad#Optimizer.zero_grad":
                cnt_optimizer_zerograd += 1
        self.assertTrue(cnt_dataloader_next == N + 1) # The final iteration will skip the loop body.
        self.assertTrue(cnt_optimizer_step == N)
        self.assertTrue(cnt_optimizer_zerograd == N)

if __name__ == '__main__':
    run_tests()
