# 数据许可、署名与图像审核

核查日期：2026-10-21；2026-10-22 复核 AWS 登记页与 CC 官方许可页。仅核查许可和现有图版，未再次下载数据、处理原始 CT/GT 或重跑评价。

## 一手来源

- [MSD / MONAI 的 AWS Open Data 登记页](https://registry.opendata.aws/msd/) 将数据许可证列为 CC BY-SA 4.0，并链接数据论文。
- [MSD 官方主页](https://medicaldecathlon.com/) 的检索摘要与登记页一致；本轮直接打开主页超时，因此不将直接抓取成功作为证据。
- [Antonelli et al., The Medical Segmentation Decathlon, Nature Communications 13, 4128 (2022)](https://doi.org/10.1038/s41467-022-30695-9) 说明数据采用 CC BY-SA 4.0，图像已去标识并转为 NIfTI。
- [Creative Commons CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 要求适当署名、许可证链接和修改说明；对改编材料适用相同或兼容许可。隐私等其他权利不由该许可自动解决。

论文自身的 CC BY 4.0 与数据的 CC BY-SA 4.0 不同，不能混用。这里记录来源要求，不替共同作者作授权决定。

## 数据引用

Simpson, A. L., Antonelli, M., Bakas, S., et al. A large annotated medical image dataset for the development and evaluation of segmentation algorithms (2019), [arXiv:1902.09063](https://arxiv.org/abs/1902.09063)。另参见 Antonelli et al. (2022)，DOI:10.1038/s41467-022-30695-9。

来源为 Medical Segmentation Decathlon Task09 Spleen，镜像由 MONAI 维护。原数据的实际下载日期未在本说明中推测；2026-10-21 是本轮许可核查日期。

## 衍生内容范围

本仓库包含数据划分/病例元数据、封存统计表和统计图。它们保持原文件字节，不发布原始体积、GT、传统或 U-Net 掩膜。`data/`、`results/`、`docs/figures/` 及 README/报告中转载的相应数据表与图按 CC BY-SA 4.0 标记；范围见 [NOTICE.md](../NOTICE.md)。保留 MSD 数据提供者署名，对原始数据不主张独占版权；项目原创软件另采用 MIT。

两张 CT/GT/传统/U-Net 三方向比较图在发布目录外待批准，不随当前公开清单发布。拟用说明为：来源 MSD Task09 Spleen，数据提供者见上述论文；closest-RAS 轴重排、CT 窗宽窗位显示、GT 最大面积切片选择、预测/GT 轮廓与图版布局为本项目衍生处理；CC BY-SA 4.0。项目贡献者使用 A/B 标识，见 [贡献说明](../CONTRIBUTORS.md)；此说明不代表 CT 图已获准公开。

本轮肉眼检查这两张图，未见姓名、出生日期、住院号、面部或本机路径；PNG 文本元数据另有自动检查。可见 case ID 是公开数据集编号，不是患者身份独立性的证明。该检查不等于可识别风险为零，最终发布还需人工确认。

## 授权与排除范围

双方已同意共同项目成果公开与 MIT 软件许可，贡献者采用 A/B 标识。数据衍生部分保留上述 CC BY-SA 4.0 来源与修改说明；两张 CT 对比图的衍生图公开授权与隐私终审仍未明确确认，因此继续排除，不影响本仓库其他已审核材料的发布。
