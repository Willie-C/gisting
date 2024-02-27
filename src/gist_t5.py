"""Gist T5 Model."""


import copy
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.models.t5 import T5ForConditionalGeneration
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    __HEAD_MASK_WARNING_MSG,
    T5Attention,
    T5Block,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerNorm,
    T5LayerSelfAttention,
    T5PreTrainedModel,
    T5Stack,
    checkpoint,
)
from transformers.utils import logging

from .generation_utils import GistGenerationMixin
from .gist_caching import GistActivations

logger = logging.get_logger(__name__)


PRETRAINED_VOCAB_SIZE = 32100


class GistT5Attention(T5Attention):
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        T5模型中注意力机制的前向传播。

        Args:
            hidden_states (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, dim)。
            mask (torch.Tensor, optional): 注意力掩码张量，形状为 (batch_size, key_length) 或 (batch_size, key_length, key_length)。
            key_value_states (torch.Tensor, optional): 如果提供，包含键和值状态的张量，用于对源句子进行注意力。
            position_bias (torch.Tensor, optional): 包含位置偏置信息的张量。
            past_key_value (tuple of torch.Tensor, optional): 上一注意力步骤的过去键和值张量。
            layer_head_mask (torch.Tensor, optional): 层头掩码张量。
            query_length (int, optional): 查询长度，如果提供。
            use_cache (bool, optional): 是否使用缓存。
            output_attentions (bool, optional): 是否输出注意力。

        Returns:
            outputs (tuple): 包含各种输出的元组，包括注意力输出、当前键值状态和位置偏置。

        Gist版本中的变化:
            1. 即使注意力块不是解码器，也支持返回present_key_value_state。
        """
        # 获取批次大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 如果提供了过去键值，更新实际序列长度
        real_seq_length = seq_length
        if past_key_value is not None:
            assert len(past_key_value) == 2, (
                "past_key_value 应该有2个过去状态: 键和值。"
                f"得到{len(past_key_value)}个过去状态"
            )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        # 获取键长度
        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """将隐藏状态投影到 (batch_size, n_heads, seq_length, dim_per_head) 形状"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """将形状重新调整为 (batch_size, seq_length, dim)"""
            return (
                states.view(batch_size, self.n_heads, -1, self.key_value_proj_dim)
                .transpose(1, 2)
                .contiguous()
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """将隐藏状态正确投影到键/值状态"""
            if key_value_states is None:
                # 自注意力
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # 交叉注意力
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # 自注意力
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # 检查`past_key_value`的`sequence_length`与提供的`key_value_states`的一致性，以支持前缀调整交叉注意力
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # 交叉注意力
                    hidden_states = past_key_value
            return hidden_states

        # 获取查询状态
        query_states = shape(
            self.q(hidden_states)
        )  # 形状为 (batch_size, n_heads, seq_length, dim_per_head)

        # 获取键/值状态
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # 计算注意力分数
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # 如果未提供位置偏置，则计算一个
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # 如果已经计算了键和值，则仅需最后一个查询位置的偏置
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            # 如果提供了掩码，则将其加入到位置偏置中
            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # 形状为 (batch_size, n_heads, seq_length, key_length)

        # 如果存在头部修剪，则将其应用到位置偏置中
        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        # 计算注意力权重
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # 形状为 (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # 形状为 (batch_size, n_heads, seq_length, key_length)

        # 如果需要，对头部进行掩码
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # 计算注意力输出
        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # 形状为 (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = None
        # 如果需要使用缓存
        if use_cache:
            if self.is_decoder:
                present_key_value_state = (key_states, value_states)
            else:
                # 在非解码器中设置 present_key_value_state，但这只应在基准测试期间发生
                warnings.warn(
                    "在非解码器中设置 present_key_value_state。"
                    "确保这仅在基准测试期间发生"
                )
                present_key_value_state = (key_states, value_states)

        # 构造输出元组
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    def compute_bias(self, query_length, key_length, device=None):
        """
        计算分箱的相对位置偏置
        """
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # 形状为 (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # 形状为 (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # 形状为 (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # 形状为 (1, num_heads, query_length, key_length)
        return values


class GistT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super(T5LayerSelfAttention, self).__init__()
        self.SelfAttention = GistT5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        前向传播函数，实现了T5层的自注意力机制。

        Args:
            hidden_states (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, dim)。
            attention_mask (torch.Tensor, optional): 注意力掩码张量，形状为 (batch_size, seq_length)。
            position_bias (torch.Tensor, optional): 位置偏置张量。
            layer_head_mask (torch.Tensor, optional): 层头掩码张量。
            past_key_value (tuple of torch.Tensor, optional): 上一注意力步骤的过去键和值张量。
            use_cache (bool, optional): 是否使用缓存。
            output_attentions (bool, optional): 是否输出注意力。

        Returns:
            outputs (tuple): 包含各种输出的元组，包括注意力输出、当前键值状态和位置偏置。

        """
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用自注意力机制
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 添加注意力输出和原始隐藏状态，并使用dropout
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[
            1:
        ]  # 添加注意力权重等信息，如果有的话
        return outputs



class GistT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super(T5Block, self).__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            GistT5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                pass
                # logger.warning(
                #     "`past_key_values` is passed to the encoder. "
                #     "Please make sure this is intended."
                # )
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"  # noqa
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[
            2:
        ]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if (
                hidden_states.dtype == torch.float16
                and torch.isinf(hidden_states).any()
            ):
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = (
                    present_key_value_state + cross_attention_outputs[1]
                )

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        # hidden-states, present_key_value_states, (self-attention position
        # bias), (self-attention weights), (cross-attention position bias),
        # (cross-attention weights)
        return outputs


class GistT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super(T5PreTrainedModel, self).__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                GistT5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gist_activations: Optional[GistActivations] = None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and "
                f"{err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or "
                f"{err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if gist_activations is not None:
            # Gist activations should only be used when computing encoder outputs.
            # The decoder will attend to encoder outputs, regardless of how
            # encoder outputs was created.
            assert (
                not self.is_decoder
            ), "Gist activations should not be passed to decoder."
            past_key_values = gist_activations.past_key_values

        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            if not self.is_decoder:
                warnings.warn(
                    "use_cache is set to True, but T5Stack is not decoder. "
                    "You must be caching a gist token."
                )

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size,
        # from_seq_length, to_seq_length] ourselves in which case we just need
        # to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # If a 2D or 3D attention mask is provided for the cross-attention we
        # need to make broadcastable to [batch_size, num_heads, seq_length,
        # seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as
                # hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device
                    )
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = (
                        encoder_extended_attention_mask.to(hidden_states.device)
                    )
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device
                    )
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device
                    )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if past_key_values is not None:
                    raise ValueError(
                        "past_key_values is incompatible with gradient checkpointing."
                    )
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. "
                        "Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                # NOTE(jayelm): `encoder_decoder_position_bias` is None the
                # first time this function is called, and the position bias arg
                # gets updated using encoder_extended_attention_mask (which is
                # the cross attention mask).
                # Then, we extract the position bias from the layer outputs and
                # save it in `encoder_decoder_position_bias` for the next call.
                # This does masking equivalent to
                # `encoder_extended_attention_mask` even though that arg is
                # unused.
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias),
            # (self-attention weights), (cross-attention position bias),
            # (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer
            # store them layer_outputs = hidden-states, key-value-states
            # (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3
                ]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things
            # on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if gist_activations is not None:
            hidden_states = torch.cat(
                [gist_activations.last_hidden_state, hidden_states], dim=1
            )
            # There should be the same number of hidden states as kv states,
            # and that number should be the seq_length plus the number of gist
            # tokens.
            if present_key_value_states is not None:
                assert hidden_states.shape[1] == present_key_value_states[0][0].shape[2]
            assert (
                hidden_states.shape[1]
                == input_ids.shape[1] + gist_activations.last_hidden_state.shape[1]
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class GistT5ForConditionalGeneration(T5ForConditionalGeneration, GistGenerationMixin):
    def __init__(self, config: T5Config):
        """初始化方法，创建一个新的GistT5ForConditionalGeneration实例。

        Args:
            config (T5Config): T5模型的配置对象。

        """
        super(T5PreTrainedModel, self).__init__(config)  # 调用父类的初始化方法
        self.model_dim = config.d_model  # 设置模型维度

        # 创建共享的嵌入层，用于将输入转换为嵌入表示
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器和解码器的配置，并根据需要修改一些参数
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器实例
        self.encoder = GistT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器实例
        self.decoder = GistT5Stack(decoder_config, self.shared)

        # 创建线性层，用于将模型的输出转换为词汇表中每个词的概率分布
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

        # 初始化模型并设定设备映射
        self.model_parallel = False
        self.device_map = None

    @torch.no_grad()
    def get_gist_activations(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        gist_token: int,
        num_gist_tokens: int,
    ) -> GistActivations:
        """计算编码器部分的gist激活。

        Args:
            input_ids (torch.LongTensor): 输入ID张量。
            attention_mask (torch.FloatTensor): 注意力掩码张量。
            gist_token (int): gist标记的ID。
            num_gist_tokens (int): gist标记的数量。

        Returns:
            GistActivations: 包含gist激活的对象。

        """
        # 在编码器上执行前向传播，获取键值激活
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,  # 需要用到缓存以获取键值激活
        )
        # 构建并返回GistActivations对象
        return GistActivations.from_model_outputs(
            model_outputs=encoder_outputs,
            input_ids=input_ids,
            gist_token=gist_token,
            num_gist_tokens=num_gist_tokens,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        cross_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        gist_activations: Optional[GistActivations] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """定义模型的前向传播逻辑。

        Args:
            input_ids (Optional[torch.LongTensor]): 输入ID张量。
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码张量。
            decoder_input_ids (Optional[torch.LongTensor]): 解码器的输入ID张量。
            decoder_attention_mask (Optional[torch.BoolTensor]): 解码器的注意力掩码张量。
            cross_attention_mask (Optional[torch.BoolTensor]): 跨注意力掩码张量。
            head_mask (Optional[torch.FloatTensor]): 头部掩码张量。
            decoder_head_mask (Optional[torch.FloatTensor]): 解码器头部掩码张量。
            cross_attn_head_mask (Optional[torch.Tensor]): 跨注意力头部掩码张量。
            encoder_outputs (Optional[Tuple[Tuple[torch.Tensor]]]): 编码器输出。
            past_key_values (Optional[Tuple[Tuple[torch.Tensor]]]): 过去的键值。
            inputs_embeds (Optional[torch.FloatTensor]): 嵌入表示张量。
            decoder_inputs_embeds (Optional[torch.FloatTensor]): 解码器的嵌入表示张量。
            labels (Optional[torch.LongTensor]): 标签张量。
            use_cache (Optional[bool]): 是否使用缓存。
            output_attentions (Optional[bool]): 是否输出注意力权重。
            output_hidden_states (Optional[bool]): 是否输出隐藏状态。
            return_dict (Optional[bool]): 是否返回字典格式的输出。
            gist_activations (Optional[GistActivations]): gist激活对象。

        Returns:
            Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]: 模型输出。

        """
                r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All
            labels set to `-100` are ignored (masked), the loss is only computed
            for labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park",
                return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the
                <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for
                you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        # 基于模型配置设置一些默认参数
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # 若头部掩码不为空，则分配给解码器的头部掩码
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # 编码输入
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                gist_activations=gist_activations,  # 添加gist激活
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # 解码器部分
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=cross_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # 线性层，将模型的输出转换为词汇表中每个词的概率分布
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # 计算损失
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            # 返回模型输出
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        # 返回字典格式的输出
        outputs = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        cross_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """准备用于生成的模型输入。

        Args:
            input_ids: 输入ID张量。
            past_key_values: 过去的键值。
            attention_mask: 注意力掩码。
            cross_attention_mask: 跨注意力掩码。
            head_mask: 头部掩码。
            decoder_head_mask: 解码器头部掩码。
            cross_attn_head_mask: 跨注意力头部掩码。
            use_cache: 是否使用缓存。
            encoder_outputs: 编码器输出。
            **kwargs: 其他关键字参数。

        Returns:
            dict: 包含各种输入参数的字典。

        """
        # 如果使用了过去的键值，则截取解码器的输入ID张量
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 构建并返回包含各种输入参数的字典
        inputs = {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "cross_attention_mask": cross_attention_mask,
            "use_cache": use_cache,
        }
        return inputs
