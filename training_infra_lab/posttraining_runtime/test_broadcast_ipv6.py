"""Exercise the patched upstream function with stubs, plus real CPU TCP rendezvous."""
import ast
from datetime import timedelta
import ipaddress
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest
from urllib.parse import urlparse

from patch_broadcast_ipv6 import patched_source

BASE = Path(__file__).resolve().parents[1] / 'research/posttraining_20261022/upstream/miles'
ORIGINAL = (BASE / 'miles/backends/training_utils/weight_update/protocols/broadcast.py').read_text()


def extract_function(source, name):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])), '<upstream-function>', 'exec')


class RendezvousTest(unittest.TestCase):
    def check_address(self, host):
        families = []
        sent = []
        captured = {}
        class Socket:
            def __init__(self, family): families.append(family)
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def bind(self, value): pass
            def getsockname(self): return ('', 47371)
        class Engine:
            def init_weights_update_group(self, *args, **kwargs):
                sent.append(args)
                return 'future'
        def init_group(**kwargs):
            captured.update(kwargs)
            return 'group'
        ns = {'ipaddress': ipaddress}
        exec(extract_function((BASE / 'miles/utils/http_utils.py').read_text(), 'wrap_ipv6'), ns)
        ns.update(ray=SimpleNamespace(_private=SimpleNamespace(services=SimpleNamespace(get_node_ip_address=lambda: host))), socket=SimpleNamespace(socket=Socket, AF_INET=socket.AF_INET, AF_INET6=socket.AF_INET6), init_process_group=init_group, async_utils=SimpleNamespace(submit=lambda x: x, wait_futures=lambda x: None))
        exec(extract_function(patched_source(ORIGINAL), 'connect_rollout_engines_from_distributed'), ns)
        self.assertEqual(ns['connect_rollout_engines_from_distributed'](SimpleNamespace(rollout_num_gpus_per_engine=1), 'test', [Engine(), Engine()], engine_gpu_counts=[1, 2]), 'group')
        parsed = urlparse(captured['init_method'])
        self.assertEqual(parsed.hostname, host.strip('[]'))
        self.assertEqual(parsed.port, 47371)
        self.assertEqual(captured['world_size'], 4)
        self.assertEqual([a[2] for a in sent], [1, 2])
        self.assertTrue(all(a[0] == host.strip('[]') for a in sent))
        self.assertEqual(families, [socket.AF_INET6 if ':' in host else socket.AF_INET])

    def test_ipv4(self): self.check_address('127.0.0.1')
    def test_ipv6(self): self.check_address('2605:340:cd51:5100:0:1fc1:77de:2a9f')
    def test_bracketed_ipv6(self): self.check_address('[::1]')
    def test_scoped_ipv6(self): self.check_address('fe80::1%eth0')

    def test_original_bug(self):
        with self.assertRaises(ValueError):
            _ = urlparse('tcp://2605:340:cd51:5100:0:1fc1:77de:2a9f:47371').port

    def test_source_guard(self):
        with self.assertRaises(ValueError): patched_source(patched_source(ORIGINAL))

    def test_real_cpu_rendezvous(self):
        import torch.distributed as dist
        for family, host in [(socket.AF_INET, '127.0.0.1'), (socket.AF_INET6, '::1')]:
            with socket.socket(family) as sock:
                sock.bind((host, 0))
                port = sock.getsockname()[1]
            url = f'tcp://[{host}]:{port}' if family == socket.AF_INET6 else f'tcp://{host}:{port}'
            iterator = dist.rendezvous(url, rank=0, world_size=1, timeout=timedelta(seconds=5))
            store, rank, size = next(iterator)
            store.set('probe', 'ok')
            self.assertEqual(store.get('probe'), b'ok')
            self.assertEqual((rank, size), (0, 1))
            del store, iterator


if __name__ == '__main__': unittest.main()
