# 版权、许可与署名

本项目由 [Kangsoolim](https://github.com/Kangsoolim) 和 [Yyyyu233](https://github.com/Yyyyu233) 共同完成，使用双方的 GitHub 账号署名。双方同意公开共同项目成果并采用 MIT 软件许可证；实际贡献见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

## 项目软件与原创文字

本仓库项目原创源码、脚本、测试、配置及原创说明文字采用 [MIT License](LICENSE)，Copyright (c) 2026 Kangsoolim and Yyyyu233。下述数据衍生材料、第三方内容及其已有权利不受该软件授权覆盖。

## MSD 数据与衍生内容

数据来源为 Medical Segmentation Decathlon Task09 Spleen，数据采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。本仓库的 `data/` 病例元数据与划分、`results/` 统计结果和训练记录、`docs/figures/` 统计图按 CC BY-SA 4.0 标记；报告和 README 中转载的相应数据、结果表与图也保留该标记。对原数据及其中可能不受版权保护的事实不主张独占权利。

这些材料包含本项目进行的数据核验、病例划分、分割评价、统计汇总、训练记录和图表排版；封存 CSV 与统计图在本次发布中未重算或重画。数据提供者、论文引用与处理说明见 [数据许可与署名](docs/DATA_LICENSE_AND_ATTRIBUTION.md)。MSD 数据许可不等同于本项目软件许可证。

原始 CT/GT、NIfTI 掩膜、模型权重未随本仓库发布。两张 CT 对比图因衍生图公开及隐私终审尚未明确批准，未纳入。

## 第三方依赖

本项目调用第三方库，不将其声明为本项目独占版权，也不以本项目 MIT 许可证重新授权第三方代码。安装和再分发时应保留所用版本要求的许可证与通知：

- [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE)
- [MONAI LICENSE](https://github.com/Project-MONAI/MONAI/blob/1.6.1/LICENSE)
- [SimpleITK LICENSE](https://github.com/SimpleITK/SimpleITK/blob/v2.5.6/LICENSE)
- NumPy、SciPy、NiBabel、Matplotlib、PyYAML 等依赖按各自发行包随附许可证执行；依赖列表见 `requirements.txt` 与 `requirements-unet.txt`。

本仓库未捆绑上述依赖的源码或二进制发行包。
