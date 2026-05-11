import pytest
import torch
from flag_gems.ops.upsample_nearest2d import upsample_nearest2d


def reference_upsample_nearest2d(input, output_size, scales_h=None, scales_w=None):
    return torch.nn.functional.interpolate(
        input, size=output_size, mode="nearest"
    )


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("C", [1, 3, 16])
@pytest.mark.parametrize("OH, OW", [(64, 64), (128, 128), (256, 128)])
@pytest.mark.parametrize("IH, IW", [(32, 32), (64, 64)])
def test_upsample_nearest2d(N, C, IH, IW, OH, OW):
    input = torch.randn(N, C, IH, IW, device="cuda", dtype=torch.float32)
    output = upsample_nearest2d(input, output_size=(OH, OW))
    expected = reference_upsample_nearest2d(input, output_size=(OH, OW))
    assert torch.allclose(output, expected, atol=1e-5)


@pytest.mark.parametrize("N", [1])
@pytest.mark.parametrize("C", [3])
@pytest.mark.parametrize("scale_h, scale_w", [(2.0, 2.0), (4.0, 2.0)])
def test_upsample_nearest2d_scales(N, C, scale_h, scale_w):
    IH, IW = 32, 32
    input = torch.randn(N, C, IH, IW, device="cuda", dtype=torch.float32)
    output = upsample_nearest2d(input, output_size=None, scales_h=scale_h, scales_w=scale_w)
    expected = torch.nn.functional.interpolate(
        input, scale_factor=(scale_h, scale_w), mode="nearest"
    )
    assert torch.allclose(output, expected, atol=1e-5)
