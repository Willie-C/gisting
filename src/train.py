# coding=utf-8

# 版权声明和许可信息

"""
Gist 训练脚本，修改自 Hugging Face 的 run_clm.py 示例。
"""

# 导入必要的库
import logging
import os

import hydra  # 配置库
import torch  # PyTorch
from datasets import DatasetDict, load_dataset  # 数据集处理
from omegaconf.dictconfig import DictConfig  # 配置管理
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    is_torch_tpu_available,
    set_seed,
)  # Hugging Face Transformers 库
from transformers.trainer_utils import get_last_checkpoint  # 训练工具
from transformers.utils import check_min_version  # 工具
from transformers.utils.versions import require_version  # 工具

# 自定义模块
from . import gist_llama, gist_t5
from .arguments import Arguments, global_setup
from .data import alpaca
from .data.utils import nested_select
from .gist_llama import DEBUG_LLAMA_CONFIG, GistLlamaForCausalLM
from .gist_t5 import GistT5ForConditionalGeneration
from .integrations import CustomWandbCallback, EvaluateFirstStepCallback
from .metrics import get_compute_metrics_fn
from .trainer_seq2seq import GistSeq2SeqTrainer

# 检查 Transformers 库的最低版本要求
check_min_version("4.28.0.dev0")

# 检查 Datasets 库的最低版本要求
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 主函数与 Hydra 配置
@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    # 设置全局配置
    args: Arguments = global_setup(args)

    # 检测最后一个检查点
    last_checkpoint = None
    if (
        os.path.isdir(args.training.output_dir)
        and args.training.do_train
        and not args.training.overwrite_output_dir
    ):
        # 获取最后一个检查点
        last_checkpoint = get_last_checkpoint(args.training.output_dir)
        # 如果没有最后一个检查点但输出目录不为空，发出警告
        if last_checkpoint is None and len(os.listdir(args.training.output_dir)) > 0:
            existing_files = os.listdir(args.training.output_dir)
            logger.warning(
                (
                    "输出目录 (%s) 已经存在且"
                    "不为空。现有文件: %s。"
                    "尽管如此仍进行训练，因为这些文件可能只是输出文件。"
                ),
                args.training.output_dir,
                str(existing_files),
            )
        # 如果有最后一个检查点且未指定从检查点恢复，则输出信息提示恢复训练
        elif (
            last_checkpoint is not None and args.training.resume_from_checkpoint is None
        ):
            logger.info(
                f"检测到检查点，从 {last_checkpoint} 恢复训练。"
                "要避免此行为，请更改"
                "`--output_dir` 或添加 `--overwrite_output_dir` 以从头开始训练。"
            )

    # 在初始化模型之前设置种子
    set_seed(args.training.seed)

    # 加载数据集
    if args.data.dataset_name == "alpaca-plus":
        # 如果数据集名称为 "alpaca-plus"，则加载数据集
        lm_datasets = load_dataset(
            "src/data/alpaca/alpaca.py",
            cache_dir=args.model.cache_dir,
        )
    else:
        raise NotImplementedError(f"未知的数据集名称 {args.data.dataset_name}")

    # 模型配置
    config_kwargs = {
        "cache_dir": args.model.cache_dir,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    if args.model.llama_debug:
        # 如果启用了 llama_debug，则使用调试配置
        if args.model.pretrained:
            raise RuntimeError("llama_debug 需要将 pretrained 设置为 False")
        config = DEBUG_LLAMA_CONFIG
    elif args.model.config_name:
        # 如果指定了 config_name，则从预训练模型名称加载配置
        config = AutoConfig.from_pretrained(args.model.config_name, **config_kwargs)
    elif args.model.model_name_or_path:
        # 否则，从预训练模型路径加载配置
        config = AutoConfig.from_pretrained(
            args.model.model_name_or_path, **config_kwargs
        )
    else:
        raise ValueError(
            "与 run_clm.py 不同，此脚本不支持从头指定模型类型。请指定 args.model.model_name_or_path，并将"
            "args.pretrained = False 以从头开始训练。"
        )

    # 分词器配置
    is_t5 = any(t in args.model.model_name_or_path.lower() for t in ("t5", "tk"))
    is_llama = any(t in args.model.model_name_or_path.lower() for t in ("llama",))

    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    if args.model.tokenizer_name:
        # 如果指定了 tokenizer_name，则从预训练分词器名称加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name, **tokenizer_kwargs
        )
    elif args.model.model_name_or_path:
        # 否则，从预训练模型路径加载分词器
        if is_llama:
            # 如果是 Llama 模型，则使用 LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(
                args.model.model_name_or_path, **tokenizer_kwargs
            )
            # 设置 LlamaTokenizer 的 pad_token 和 padding_side
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        else:
            # 否则，使用 AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.model.model_name_or_path, **tokenizer_kwargs
            )
    else:
        raise ValueError(
            "您正在从头实例化一个新的分词器。此脚本不支持此操作。"
            "您可以在另一个脚本中执行此操作，并将其保存，然后使用 --tokenizer_name 从此处加载。"
        )

    # 模型实例化
    if is_t5:
        # 如果是 T5 模型，则使用 GistT5ForConditionalGeneration
        model_cls = GistT5ForConditionalGeneration
    elif is_llama:
        # 如果是 Llama 模型，则使用 GistLlamaForCausalLM
        model_cls = GistLlamaForCausalLM
    else:
        raise ValueError(f"不支持模型类型 {args.model.model_name_or_path}")
    if args.model.pretrained:
        # 如果是预训练模型，则从预训练模型加载模型
        model = model_cls.from_pretrained(
            args.model.model_name_or_path,
            from_tf=bool(".ckpt" in args.model.model_name_or_path),
            config=config,
            cache_dir=args.model.cache_dir,
            revision=args.model.model_revision,
            use_auth_token=True if args.model.use_auth_token else None,
        )
    else:
        # 否则，实例化一个新的模型
        model = model_cls(config)

    # Gist 令牌处理
    if is_t5 and len(tokenizer) == gist_t5.PRETRAINED_VOCAB_SIZE + 1:
        # 如果是 T5 模型且分词器的词汇表大小与预定义的大小相符，则验证权重矩阵的形状
        assert model.shared.weight.shape[0] == gist_t5.PRETRAINED_VOCAB_SIZE + 1
    elif is_llama and len(tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 1:
        # 如果是 Llama 模型且分词器的词汇表大小与预定义的大小相符，则验证权重矩阵的形状
        assert (
            model.model.embed_tokens.weight.shape[0]
            == gist_llama.PRETRAINED_VOCAB_SIZE + 1
        )
        assert model.lm_head.weight.shape[0] == gist_llama.PRETRAINED_VOCAB_SIZE + 1
    else:
        # 否则，初始化 Gist 令牌
        tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
        model.resize_token_embeddings(len(tokenizer))
        # 将新单词嵌入设置为现有单词嵌入的平均值
        if args.model.pretrained:
            with torch.no_grad():
                if is_t5:
                    model.shared.weight[-1] = model.shared.weight[:-1].mean(0)
                elif is_llama:
                    model.model.embed_tokens.weight[
                        -1
                    ] = model.model.embed_tokens.weight[:-1].mean(0)
                    model.lm_head.weight[-1] = model.lm_head.weight[:-1].mean(0)
                else:
                    raise ValueError(
                        f"不支持模型类型 {args.model.model_name_or_path}"
                    )
    gist_token = tokenizer.additional_special_tokens_ids[-1]

    if args.training.do_train:
        # 如果进行训练，加载训练数据集
        if "train" not in lm_datasets:
            raise ValueError("--do_train 需要一个训练数据集")
        train_dataset = lm_datasets["train"]
        if args.data.max_train_samples is not None:
            # 如果指定了最大训练样本数，则截取数据集
            max_train_samples = min(len(train_dataset), args.data.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if args.training.do_eval:
        # 如果进行评估，加载验证数据集
        validation_splits = [
            split for split in lm_datasets if split.startswith("validation")
        ]
        if not validation_splits:
            raise ValueError(
                "--do_eval 需要至少一个以 `validation` 开头的验证数据集"
            )
        eval_dataset = DatasetDict(
            # 修剪 "validation-" 前缀
            {split[11:]: lm_datasets[split] for split in validation_splits}
        )
        # 在截取样本之前，（确定性地）对验证集进行洗牌
        eval_dataset = eval_dataset.shuffle(seed=42)
        if args.data.max_eval_samples is not None:
            # 如果指定了最大评估样本数，则截取数据集
            eval_dataset = nested_select(
                eval_dataset,
                args.data.max_eval_samples,
            )

        # 获取计算指标函数
        compute_metrics = get_compute_metrics_fn(
            gist_token=gist_token, tokenizer=tokenizer, args=args
        )

    if is_t5:
        # 如果是 T5 模型，则使用 alpaca.collator.DataCollatorForAlpaca
        data_collator = alpaca.collator.DataCollatorForAlpaca(
            tokenizer,
            model=model,
            padding="longest",
            # 选择确保 <1% 的示例被截断
            # 有关长度统计信息，请参见 data/alpaca_plus/length_stats.txt
            max_source_length=128,
            max_target_length=256,
            # 人类评估示例较长
            max_source_length_human=384,
            max_target_length_human=384,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if args.training.fp16 else None,
            gist_condition=args.training.gist.condition,
            num_gist_tokens=args.training.gist.num_gist_tokens,
            gist_token=gist_token,
            pad_token=tokenizer.pad_token_id,
            add_gist_token=args.training.gist.add_gist_token,
        )
    elif is_llama:
        # 如果是 Llama 模型，则使用 alpaca.collator.DataCollatorForAlpacaCLM
        data_collator = alpaca.collator.DataCollatorForAlpacaCLM(
            tokenizer,
            # 选择确保 <1% 的示例被截断
            # 有关长度统计信息，请参见 data/alpaca_plus/length_stats.txt
            max_length=256 + 256,  # source=256; target=256
            # 人类评估示例较长
            max_length_human=384 + 384,  # source=384; target=384
            gist_condition=args.training.gist.condition,
            num_gist_tokens=args.training.gist.num_gist_tokens,
            gist_token=gist_token,
            pad_token=tokenizer.pad_token_id,
            check_correctness=True,
        )
    else:
        assert False, "应该是 is_llama 或 is_t5"

    # 初始化训练器
    custom_callbacks = []
    if args.wandb.log:
        custom_callbacks.append(CustomWandbCallback(args))
    if args.training.evaluate_before_train:
        custom_callbacks.append(EvaluateFirstStepCallback())

    trainer = GistSeq2SeqTrainer
