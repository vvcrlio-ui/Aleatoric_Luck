# NKGRID

重复抽样的N×K预测实验，包含SMR与FFCWS应用。

## 单命令启动

Linux本地（默认FFC GPA、dev规模）：

```bash
bash run.sh local
```

BMRC集群（默认dev，自动准备环境、提交计划生成作业，完成后提交计算与汇总链）：

```bash
# 将 YOUR_PROJECT_ACCOUNT 替换为你获准使用的项目账户（无默认值）
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT
```

先查看启动配置，不安装、不读取实验数据、不提交作业：

```bash
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT --preset production --dry-run
```

首次使用需要可读的私有数据和Adapter生成的schema/ARD。已有其他位置的输入用 `--schema` 指定，不要求复制或覆盖原文件。详见 [启动与维护说明](launch/README.md)，其中包含production、恢复、Windows/WSL、环境和验收边界。

研究设计：[NK_Grid](NK_Grid/README.md) · [SMR](SMR/README.md) · [FFCWS](FFCWS/README.md)。
