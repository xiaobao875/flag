import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = shape
    input_tensor = torch.randn(
        (batch, input_c, input_h, input_w), device=device, dtype=dtype
    )
    weight = torch.randn(
        (out_c, input_c // groups, kernel_h, kernel_w), device=device, dtype=dtype
    )
    yield input_tensor, weight, [padding, padding], [stride, stride], [
        1,
        1,
    ], groups, False, False, False


class CudnnConv2DBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (32, 64, 128, 128, 32, 3, 3, 1, 2, 1),
        (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
        (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
        (16, 32, 24, 24, 24, 3, 3, 2, 2, 2),
        (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
    ]

    def set_more_shapes(self):
        return []

    def set_shapes(self, *args, **kwargs):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from _input_fn(shape, dtype, self.device)


@pytest.mark.cudnn_convolution
def test_cudnn_convolution():
    torch.backends.cudnn.allow_tf32 = False
    bench = CudnnConv2DBenchmark(
        op_name="cudnn_convolution",
        torch_op=torch.ops.aten.cudnn_convolution.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
