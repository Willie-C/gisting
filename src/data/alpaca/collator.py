import logging  # 引入日志记录模块
from collections import defaultdict  # 引入默认字典模块
from dataclasses import dataclass  # 引入数据类模块
from typing import Any, Optional, Union  # 引入类型提示模块

import torch  # 引入 PyTorch 库
from torch.nn.utils.rnn import pad_sequence  # 引入序列填充函数
from transformers import PreTrainedTokenizerBase  # 引入预训练分词器基类
from transformers.data.data_collator import PaddingStrategy  # 引入填充策略
from ...utils import first_mismatch  # 从自定义工具模块中导入函数
from .. import gist  # 从自定义模块中导入 gist 相关函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

@dataclass
class DataCollatorForAlpaca:
    """
    Alpaca 数据收集器类，用于处理模型的输入数据。
    """

    tokenizer: PreTrainedTokenizerBase  # 预训练分词器
    model: Optional[Any] = None  # 模型，可选
    padding: Union[bool, str, PaddingStrategy] = True  # 填充策略，默认为 True
    max_source_length: Optional[int] = None  # 源文本的最大长度，可选
    max_target_length: Optional[int] = None  # 目标文本的最大长度，可选
    max_source_length_human: Optional[int] = None  # 人类示例的源文本最大长度，可选
    max_target_length_human: Optional[int] = None  # 人类示例的目标文本最大长度，可选
    pad_to_multiple_of: Optional[int] = None  # 填充到的倍数，可选
    label_pad_token_id: int = -100  # 标签的填充标记ID，默认为 -100
    return_tensors: str = "pt"  # 返回的张量类型，默认为 "pt" (PyTorch)
    gist_token: int = 32100  # 关键片段标记的ID，默认为 32100
    pad_token: int = 0  # 填充标记的ID，默认为 0
    add_gist_token: bool = True  # 是否在输入中添加关键片段标记，默认为 True
    gist_condition: str = "gist"  # 控制关键片段遮罩的条件，默认为 "gist"
    num_gist_tokens: int = 10  # 添加到输入中的关键片段标记数量，默认为 10

    def __post_init__(self):
        """
        对实例进行后期初始化。
        """
        if self.max_source_length_human is None:
            self.max_source_length_human = self.max_source_length
        if self.max_target_length_human is None:
            self.max_target_length_human = self.max_target_length

    def __call__(self, batch, return_tensors=None):
        """
        将批次数据转换为模型输入格式的方法。
        """
        if any("human" in instance["split"] for instance in batch):
            # 使用人类示例的最大长度
            max_source_length = self.max_source_length_human
            max_target_length = self.max_target_length_human
        else:
            max_source_length = self.max_source_length
            max_target_length = self.max_target_length

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []  # 存储源文本
        for instance in batch:
            if not self.add_gist_token:
                # 在分词期间稍后添加关键片段标记
                maybe_gist_str = ""
            else:
                maybe_gist_str = "<GIST>" * self.num_gist_tokens  # 重复添加关键片段标记

            if instance["input"]:
                # 如果有输入文本，则构建包含输入的源文本
                source = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nInput: {instance['input']}"
            else:
                # 如果没有输入文本，则只包含指令
                source = f"Instruction: {instance['instruction']}\n{maybe_gist_str}"

            # 分词源文本并截断到指定长度
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= max_source_length:
                tokenized_source = tokenized_source[:-1]  # 去除 "</s>" 标记
            else:
                tokenized_source = tokenized_source[:max_source_length]
            sources.append(self.tokenizer.decode(tokenized_source))

        # 使用分词器处理源文本
        model_inputs = self.tokenizer(
            sources,
            max_length=max_source_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # 分词标签
        labels = [instance["output"] for instance in batch]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                labels,
                max_length=max_target_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        label_mask = labels["attention_mask"].bool()
        model_inputs["labels"] = labels["input_ids"].masked_fill(
            ~label_mask, self.label_pad_token_id
        )

        # 准备解码器输入IDs
        if self.model is not None and hasattr(
            self.model, "prepare_decoder_input_ids_from_labels"
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=model_inputs["labels"]
            )
            model_inputs["decoder_input_ids"] = decoder_input_ids

        # 修改注意力遮罩
        if self.gist_condition == "pos_control" or not self.add_gist_token:
            # 保持不变，仅设置交叉注意力遮罩
            model_inputs["cross_attention_mask"] = model_inputs["attention_mask"]
        elif self.gist_condition == "gist":
            # 使用 gist.make_gist_mask 生成关键片段遮罩
            model_inputs["attention_mask"] = gist.make_gist_mask(
                model_inputs["input_ids"],
                self.gist_token,
                pad_token=self.pad_token,
            ).squeeze(
                1
            )  # T5代码库期望3D遮罩，因此挤压维度1
            # 解码器交叉注意力不能看到第一个关键片段标记之前的内容
            model_inputs["cross_attention_mask"] = gist.make_mask_pre_first_gist(
                model_inputs["input_ids"],
                self.gist_token,
                pad_token=self.pad_token,
            )
        elif self.gist_condition == "neg_control":
            # 使用 gist.make_neg_control_mask 生成负控制遮罩
            model_inputs["attention_mask"] = gist.make_neg_control_mask(
                model_inputs["input_ids"],
                self.gist_token,
                pad_token=self.pad_token,
            ).squeeze(
                1
            )  # T5代码库期望3D遮罩，因此挤压维度1
            # 解码器交叉注意力不能看到（包括）任何关键片段标记之前的内容
            model_inputs["cross_attention_mask"] = 1 - (
                gist.make_mask_post_last_gist(
                    model_inputs["input_ids"],
                    self.gist_token,
                    pad_token=self.pad_token,
                )
            )
        else:
            raise ValueError(f"Invalid gist_condition: {self.gist_condition}")

        return model_inputs

@dataclass
class DataCollatorForAlpacaCLM:
    """
    用于仅解码器模型的数据收集器类。执行左填充。
    """

    tokenizer: PreTrainedTokenizerBase  # 预训练分词器
    max_length: Optional[int] = None  # 最大长度，可选
    max_length_human: Optional[int] = None  # 人类示例的最大长度，可选
    label_pad_token_id: int = -100  # 标签填充标记ID，默认为 -100
    return_tensors: str = "pt"  # 返回的张量类型，默认为 "pt" (PyTorch)
    gist_token: int = 50257  # 关键片段标记的ID，默认为 50257
    pad_token: int = 0  # 填充标记的ID，默认为 0
    add_gist_token: bool = True  # 是否在输入中添加关键片段标记，默认为 True
    gist_condition: str = "gist"  # 控制关键片段遮罩的条件，默认为 "gist"
    num_gist_tokens: int = 10  # 添加到输入中的关键片段标记数量，默认为 10
    check_correctness: bool = False  # 是否检查正确性，默认为 False

    def __post_init__(self):
        """
        对实例进行后期初始化。
        """
        if self.max_length_human is None:
            self.max_length_human = self.max_length

    def __call__(self, batch, return_tensors=None):
        """
        将批次数据转换为模型输入格式的方法。
        """
        if any("human" in instance["split"] for instance in batch):
            # 使用人类示例的最大长度
            max_length = self.max_length_human
        else:
            max_length = self.max_length

        if return_tensors is None:
            return_tensors = self.return_tensors

        model_inputs = defaultdict(list)
        for instance in batch:
            if not self.add_gist_token:
                # 在分词期间稍后添加关键片段标记
                maybe_gist_str = ""
            else:
                maybe_gist_str = " ".join(
                    ["<GIST>" for _ in range(self.num_gist_tokens)]
                )  # 重复添加关键片段标记

            if instance["input"]:
                # 如果有输入文本，则构建包含输入的源文本
                prompt = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nInput: {instance['input']}\nOutput:"
            else:
                # 如果没有输入文本，则只包含指令
                prompt = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nOutput:"

            # 提示和完成文本分词
            tokenized_prompt = self.tokenizer(prompt)["input_ids"]
            tokenized_completion = self.tokenizer(completion, add_special_tokens=False)[
                "input_ids"
            ] + [self.tokenizer.eos_token_id]

            # 检查分词后的提示和完成是否与一起分词的结果相同
            if self.check_correctness:
                combined = tokenized_prompt + tokenized_completion
                real = self.tokenizer(prompt + " " + completion)["input_ids"] + [
                    self.tokenizer.eos_token_id
                ]
                if combined != real:
                    logger.warning(
                        (
                            "Tokenizing prompt/completion separately gave different "
                            "results. This is usually because the output is empty. "
                            "First mismatch location: %s. Source: %s",
                        ),
                        str(first_mismatch(combined, real)),
                        self.tokenizer.decode(combined),
                    )
                    continue

            tokenized_source = tokenized_prompt + tokenized_completion
            labels = [self.label_pad_token_id] * len(
                tokenized_prompt
            ) + tokenized_completion

            # 如果源文本长度超过最大长度，则截断
            if len(tokenized_source) > max_length:
                to_trim = len(tokenized_source) - max_length
                tokenized_source = tokenized_source[:-to_trim]
                labels = labels[:-to_trim]
                logger.warning(
                    "Truncating source on right from %d to %d tokens. Result: %s",
                    max_length + to_trim,
                    max_length,
                    self.tokenizer.decode(tokenized_source),
                )
                if to_trim >= len(tokenized_completion):
                    logger.warning(
                        "^^^ The above truncated the entire "
                        "completion! Skipping loading this batch element."
                    )
                    continue

            # 将数据添加到模型输入中
            model_inputs["input_ids"].append(tokenized_source)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1 for _ in tokenized_source])

            model_inputs["prompt_input_ids"].append(tokenized_prompt)
            model_inputs["prompt_attention_mask"].append([1 for _ in tokenized_prompt])

            model_inputs["completion_input_ids"].append(tokenized_completion)
            model_inputs["completion_attention_mask"].append(
                [1 for _ in tokenized_completion]
            )

        # 对输入进行左填充并转换为张量
        for key, value in model_inputs.items():
            if key == "labels":
                pad_token_id = self.label_pad_token_id
            else:
                pad_token_id = self.tokenizer.pad_token_id
            value_tensors = [torch.tensor(v[::-1]) for v in value]
            model_inputs[key] = torch.fliplr(
                pad_sequence(
                    value_tensors,
                    batch_first=True,
                    padding_value=pad_token_id,
                )
            )

        # 构建关键片段遮罩
        if self.gist_condition == "gist":
            gist_fn = gist.make_gist_mask
        elif self.gist_condition == "neg_control":
            gist_fn = gist.make_neg_control_mask
        elif self.gist_condition == "pos_control":
            gist_fn = gist.make_pos_control_mask
        else:
            raise ValueError(f"Unknown gist condition {self.gist_condition}")
        model_inputs["attention_mask_gist"] = gist_fn(
            model_inputs["input_ids"],
            self.gist_token,
        )
        model_inputs["prompt_attention_mask_gist"] = gist_fn(
            model_inputs["prompt_input_ids"],
            self.gist_token,
        )

        return model_inputs
