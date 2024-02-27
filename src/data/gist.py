"""Utilities for gist mask generation."""
from typing import Optional, Tuple

import torch


def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    """从右到左的累积和。

    Args:
        x: 形状为 (batch_size, seq_len) 的张量
    Returns:
        形状为 (batch_size, seq_len) 的张量，其中每个元素是其右侧所有元素的和。
    """
    return x + torch.sum(x, dim=-1, keepdims=True) - torch.cumsum(x, dim=-1)


def make_mask_pre_first_gist(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """返回一个遮罩，其中所有位于第一个关键片段标记之前的标记都被屏蔽。

    Args:
        inputs: 形状为 (batch_size, seq_len) 的输入标记数组。
        gist_token: 关键片段标记的整数ID。
        pad_token: 如果提供，则屏蔽 inputs == pad_token 的位置。
        dtype: 遮罩的数据类型，默认为 int64。
    Returns:
        所请求的遮罩。
    """
    mask = (inputs == gist_token).cumsum(-1) >= 1
    if pad_token is not None:
        mask = mask & (inputs != pad_token)
    return mask.type(dtype)


def make_mask_post_last_gist(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """返回一个遮罩，其中所有位于最后一个关键片段标记之后的标记都被屏蔽。

    Args:
        inputs: 形状为 (batch_size, seq_len) 的输入标记数组。
        gist_token: 关键片段标记的整数ID。
        pad_token: 如果提供，则屏蔽 inputs == pad_token 的位置。
        dtype: 遮罩的数据类型，默认为 int64。
    Returns:
        所请求的遮罩。
    """
    mask = reverse_cumsum(inputs == gist_token) >= 1
    if pad_token is not None:
        mask = mask & (inputs != pad_token)
    return mask.type(dtype)


def make_gist_mask(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Creates a 4D gist mask.
    Here, tokens after the last gist cannot attend to tokens prior to the first
    gist.
    Additionally, tokens *before* the last gist cannot attend to tokens *after*
    the last gist.

    Example, where G is the gist token:

      a b c G d
    a 1 1 1 1 0
    b 1 1 1 1 0
    c 1 1 1 1 0
    G 1 1 1 1 0
    d 0 0 0 1 1
    创建一个四维的关键片段遮罩。

    Args:
        inputs: 形状为 (batch_size, seq_len) 的输入标记数组。
        gist_token: 关键片段标记的整数ID。
        pad_token: 如果提供，则屏蔽 inputs == pad_token 的位置。
        dtype: 遮罩的数据类型，默认为 int64。
    Returns:
        所请求的形状为 (batch_size, 1, seq_len, seq_len) 的遮罩。
    """
    pre_gist_mask = make_mask_post_last_gist(inputs, gist_token, dtype=torch.bool)[
        :, None, None
    ]
    post_gist_mask = make_mask_pre_first_gist(inputs, gist_token, dtype=torch.bool)[
        :, None, None
    ]
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))

    mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)

    has_gist = (inputs == gist_token).any(-1)[:, None, None, None]
    mask = torch.where(has_gist, mask, True)
    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


def make_neg_control_mask(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
):
    """Creates a 4D neg control mask.
    Here, tokens after the last gist cannot attend to any gist tokens (or prior).

    Example, where G is the gist token:

      a b c G d
    a 1 1 1 1 0
    b 1 1 1 1 0
    c 1 1 1 1 0
    G 1 1 1 1 0
    d 0 0 0 0 1
    创建一个四维的负控制遮罩。

    Args:
        inputs: 形状为 (batch_size, seq_len) 的输入标记数组。
        gist_token: 关键片段标记的整数ID。
        pad_token: 如果提供，则屏蔽 inputs == pad_token 的位置。
        dtype: 遮罩的数据类型，默认为 int64。
    Returns:
        所请求的形状为 (batch_size, 1, seq_len, seq_len) 的遮罩。
    """
    pre_gist_mask = make_mask_post_last_gist(inputs, gist_token, dtype=torch.bool)[
        :, None, None
    ]
    post_gist_mask = torch.logical_not(pre_gist_mask)
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))

    mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)

    has_gist = (inputs == gist_token).any(-1)[:, None, None, None]
    mask = torch.where(has_gist, mask, True)

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


def make_pos_control_mask(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
):
    """创建一个四维的正控制遮罩。

    Args:
        inputs: 形状为 (batch_size, seq_len) 的输入标记数组。
        gist_token: 关键片段标记的整数ID。
        pad_token: 如果提供，则屏蔽 inputs == pad_token 的位置。
        dtype: 遮罩的数据类型，默认为 int64。
    Returns:
        所请求的形状为 (batch_size, 1, seq_len, seq_len) 的遮罩。
    """
    del gist_token
    batch_size, seq_len = inputs.shape
    mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool)

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


def get_gist_index(
    input_ids: torch.Tensor, gist_token: int, raise_if_no_tokens: bool = False
) -> Tuple[Optional[int], Optional[int]]:
    """查找 input_ids 中关键片段范围的起始和结束。

    Args:
        input_ids: 输入id张量。
        gist_token: 关键片段标记的值。
        raise_if_no_tokens: 如果没有关键片段标记，则引发错误。

    Returns:
        如果存在的话，关键片段的起始和结束（不包括结束），否则如果 raise_if_no_tokens 为 False 则返回 (None, None)（如果为 True，则引发错误）。

    Raises:
        RuntimeError: 如果输入中的关键片段标记不是连续的范围。
        ValueError: 如果找不到关键片段标记并且 raise_if_no_tokens 为 True。
    """
    gist_indices = (input_ids == gist_token).nonzero().squeeze(-1)
    if len(gist_indices) == 0:
        if raise_if_no_tokens:
            raise ValueError(f"在 {input_ids} 中找不到关键片段标记 {gist_token}")
        return (None, None)
    _assert_continguous_span(gist_indices)
    return (gist_indices[0].item(), gist_indices[-1].item() + 1)


def get_first_pad_index(input_ids: torch.Tensor, pad_token: int) -> int:
    """找到 input_ids 中第一个 pad token 的索引。

    Args:
        input_ids: 输入id张量。
        pad_token: pad token 的值。

    Returns:
        pad token 的索引（如果存在），否则返回 len(input_ids)。
    """
    pad_indices = (input_ids == pad_token).nonzero()
    if len(pad_indices) == 0:
        return len(input_ids)
    return pad_indices[0].item()


def _assert_continguous_span(gist_indices: torch.Tensor):
    """断言关键片段索引形成一个连续的范围。"""
    gist_start = gist_indices[0]
    gist_indices_arange = torch.arange(
        start=gist_start,
        end=gist_start + len(gist_indices),
        device=gist_indices.device,
    )
    if not (gist_indices == gist_indices_arange).all():
        raise RuntimeError(f"关键片段标记不形成连续的范围: {gist_indices}") 
