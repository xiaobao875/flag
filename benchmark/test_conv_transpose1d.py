import pytest
import torch

import flag_gems

from . import base


def _input_fn(shape, dtype, device):
    batch, input_c, input_l, out_c, kernel, stride, padding, groups = shape
    inp = torch.randn((batch, input_c, input_l), device=device, dtype=dtype)
    weight = torch.randn((input_c, out_c // groups, kernel), device=device, dtype=dtype)
    yield inp, weight, None, stride, padding, 0, groups


class ConvTranspose1DBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (32, 64, 128, 64, 3, 1, 0, 1),
        (64, 48, 256, 128, 5, 2, 2, 1),
        (16, 24, 512, 96, 7, 1, 3, 1),
        (8, 16, 1024, 32, 3, 2, 1, 2),
        (4, 8, 2048, 16, 5, 1, 2, 1),
    ]

    def set_more_shapes(self):
        return []

    def set_shapes(self, *args, **kwargs):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from _input_fn(shape, dtype, self.device)


@pytest.mark.conv_transpose1d
def test_conv_transpose1d():
    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose1DBenchmark(
        op_name="conv_transpose1d",
        torch_op=torch.nn.functional.conv_transpose1d,
        dtypes=[torch.float16, torch.float32],
    )
    bench.set_gems(flag_gems.conv_transpose1d)
    bench.run()
