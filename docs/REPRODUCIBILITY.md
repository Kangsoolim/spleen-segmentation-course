# 公开版复现与验证边界

本版本从现有审核材料选择性复制，没有原开发仓库的 .git、历史提交或 cherry-pick。准备轮次未初始化 Git、训练、推理或重跑独立测试。源码中的固定历史哈希仅作为来源校验常量保留，不表示这些 Git 对象随包提供。

## 可运行内容

README 给出数据获取、传统 seed/ROI 交互、半自动分割、自动 U-Net 推理和通用三维评价的 CLI。所有命令从项目根目录运行。正式使用需要另行取得原始 MSD 数据；当前包不包含任何 CT/GT 或 NIfTI 掩膜。桌面 seed/ROI 选择依赖可交互 Matplotlib 后端。

`python -B -m unittest discover -s tests -t . -v` 使用合成数组及临时合成 NIfTI，不加载真实病例、不调用模型训练或真实推理。公开测试保留空间方向、非法输入、指标定义、选模规则和完整批次调度检查。去除了依赖旧 Git 对象、未公开 Epoch 9 checkpoint 和优化器更新的三项历史测试；从原诊断脚本只提取 tensor-aware equality 到 tests/helpers.py，避免整条诊断脚本依赖链。未通过跳过失败断言来宣称原完整测试集通过。

## 训练入口限制

`src/unet_pipeline.py` 与 `scripts/train_unet.py` 保持原始字节。训练会读取当前 Git HEAD 和状态写入来源信息，因此必须在已有首个提交的 Git 工作副本中，按 README 从项目根目录运行。本次发布验证不执行训练或真实病例推理。

最终 Stage 7 配置体现从 Epoch 14 严格续训的设置。单独从头运行该配置只能训练同架构模型，不能重现原分阶段轨迹；Epoch 17 模型需要先前训练/分叉链、相同源码字节、环境、冻结划分和 checkpoint。当前不提供阶段 checkpoint，不绕过源码、环境或签名校验。

## 最终推理限制

公开 `predict_unet.py` 读取完整 schema 1/2 checkpoint，包括 model_state_dict 与 config，默认使用 threshold=0.5、batch size=4、FP32。当前不提供 Epoch 17 权重。用户需自行获得该 checkpoint 或训练自己的模型；自训不能保证相同结果。裸 state_dict 还需要另行适配加载接口，本轮没有导出权重。

原 Stage 8 一次性冻结测试驱动依赖旧 Git 对象、传统 NIfTI 掩膜、保全清单和完整本地审计日志，未纳入公开版本。通用评价函数和 CLI 已保留；这不意味着可以用它重跑封存测试或调优同一六例。

## 可核对的结果

`results/unet_stage8_final_test/` 的全部 CSV 均与原项目封存文件字节一致。`results/final_test_evaluation/` 保存传统方法结果表；`results/unet_training_history/` 保存 Stage 4–7 原 CSV。没有重算、替换或删除任何测试病例。公开文件清单中的 SHA-256 可用于下载后校验。

CT 对比图未纳入本发布目录；六例完整量化结果和成功/失败分析仍保留。训练曲线、配对指标图、体积图从已有成果直接复制，不生成新分数。数据和权重仍需另行取得；单元测试不验证真实病例全链路复现，也不代替 CT 衍生图的公开授权与隐私终审。
