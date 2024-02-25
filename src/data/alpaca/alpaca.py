"""这段代码定义了一个名为AlpacaPlus的数据集构建器，用于组合Alpaca和Self-Instruct数据集。"""
# 导入必要的库
import json  # 用于处理 JSON 数据
import datasets  # 使用 Hugging Face datasets 库构建数据集
from datasets.splits import NamedSplit  # 使用 NamedSplit 来定义特定命名的数据拆分

# 获取 logger 实例
logger = datasets.logging.get_logger(__name__)

# 定义 AlpacaConfig 类，用于配置 AlpacaPlus 数据集构建器的参数
class AlpacaConfig(datasets.BuilderConfig):
    def __init__(
        self,
        *args,
        train_file=None,
        validation_seen_file=None,
        validation_unseen_file=None,
        validation_human_file=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # 定义训练集、验证集（已见、未见和人类）文件的路径参数
        self.train_file: str = train_file
        self.validation_seen_file: str = validation_seen_file
        self.validation_unseen_file: str = validation_unseen_file
        self.validation_human_file: str = validation_human_file

# 定义 AlpacaPlus 类，用于构建 AlpacaPlus 数据集
class AlpacaPlus(datasets.GeneratorBasedBuilder):
    """AlpacaPlus Dataset."""

    # 定义数据集的版本信息
    VERSION = datasets.Version("1.0.1")
    # 设置构建器配置类和默认配置
    BUILDER_CONFIG_CLASS = AlpacaConfig
    BUILDER_CONFIGS = [
        AlpacaConfig(
            name="default",
            train_file="./data/alpaca_plus/alpaca_plus_train.json",
            validation_seen_file="./data/alpaca_plus/alpaca_plus_validation_seen.json",
            validation_unseen_file="./data/alpaca_plus/alpaca_plus_validation_unseen.json",  # noqa
            validation_human_file="./data/alpaca_plus/alpaca_plus_validation_human.json",  # noqa
            description="Default config for Alpaca",
        ),
    ]
    DEFAULT_CONFIG_NAME = "default"

    # 定义数据集的信息，包括描述和特征
    def _info(self):
        return datasets.DatasetInfo(
            description="Alpaca Data",
            features=datasets.Features(
                {
                    "instruction": datasets.Value("string"),  # 指示信息
                    "input": datasets.Value("string"),  # 输入信息
                    "output": datasets.Value("string"),  # 输出信息
                    "source": datasets.Value("string"),  # 来源信息
                    "split": datasets.Value("string"),  # 数据集拆分信息
                }
            ),
            supervised_keys=None,  # 没有监督学习的键，因为这是一个生成器数据集
        )

    # 定义数据集的拆分方式，返回训练集和验证集的 SplitGenerator
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        del dl_manager  # 不使用下载管理器
        # 返回训练集和验证集（已见、未见和人类）的 SplitGenerator
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # 训练集
                gen_kwargs={
                    "path": self.config.train_file,  # 训练集文件路径
                    "split": "train",  # 拆分类型为训练集
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_seen"),  # 已见验证集
                gen_kwargs={
                    "path": self.config.validation_seen_file,  # 已见验证集文件路径
                    "split": "validation_seen",  # 拆分类型为已见验证集
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_human"),  # 人类验证集
                gen_kwargs={
                    "path": self.config.validation_human_file,  # 人类验证集文件路径
                    "split": "validation_human",  # 拆分类型为人类验证集
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_unseen"),  # 未见验证集
                gen_kwargs={
                    "path": self.config.validation_unseen_file,  # 未见验证集文件路径
                    "split": "validation_unseen",  # 拆分类型为未见验证集
                },
            ),
        ]

    # 定义如何生成数据集的示例
    def _generate_examples(
        self,
        path: str,  # 文件路径
        split: str,  # 数据集拆分类型
    ):
        """Yields examples."""
        # 记录正在生成的数据集拆分类型和文件路径
        logger.info(f"Generating {split} tasks from = {path}")
        # 打开文件并加载 JSON 数据
        with open(path, encoding="utf-8") as split_f:
            task_json = json.load(split_f)
            # 遍历 JSON 数据中的每个示例并生成数据集示例
            for idx, instance in enumerate(task_json):
                instance["split"] = split  # 添加拆分类型信息到示例中
                # 以生成器的形式返回数据集示例
                yield f"alpaca_{split}_{idx}", instance
