# Nion UHEMRS 像差校正：神经网络训练基线

> **环境提醒：本页中的 PyTorch 安装、数据集生成和训练命令只供容量充足的工作站使用，不在当前这台 Mac 上执行。** 当前 Mac 不安装 PyTorch，也不保留训练虚拟环境、数据集或权重。

## 目标与边界

目标是从二维零损峰光斑估计当前模型的 20 项有效像差系数，并输出使残差趋近于零的系数补偿量 `−Eij`。这不是 Nion 线圈、电压或 multipole 控制值：真实仪器上的控制组合还需要在本机测得“执行器改变量 → 像差系数改变量”的响应矩阵，并经过操作员审核。当前代码不会连接 Nion Swift、采集相机或写入任何仪器控制。

照片中的方案使用 `101 × 101 × 3` 输入、AlexNet 风格的五个特征块、`9216 → 2048 → 512 → 20` 回归头。本仓库实现相同量级的回归头，但灰度单图使用一个通道；推荐的已知探测对使用两个通道。

最终闭环应分成两层：`图像 I → 有效像差补偿 ΔE → Nion 控制改变量 Δc`。第一层可以先用模拟数据训练；第二层必须在目标 UHEMRS 上测量响应矩阵 `ΔE = JΔc`。仓库实现了第一层和“读取经审核的响应矩阵后离线计算第二层建议”的工具，但不虚构 Nion 控制名、单位、极性或安全量程，也没有仪器写入代码。

## 为什么推荐两张图而不是一张

当前前向模型积分掉了入射孔径坐标 `u`，孔径采样关于 `u=0` 对称。因此，把所有 `u` 的奇次幂项同时反号，相当于变量替换 `u → −u`，得到完全相同的单张图。严格简并的项为：

`D10, D11, D30, D12, D31, D13, D50, D32, D14`

所以单图网络至多识别一个等价类，不能保证给出这些项的正确执行方向。`single` 模式仍可用于复现照片中的基线，并用对称感知损失训练；推理结果会同时报告另一组等价答案。

推荐 `probe-pair`：先记录初始图，再施加一个**已知且可逆**的小探测量（默认模型坐标中的 `D10 +12 meV`）记录第二张图。这个第二观测打破上述符号对称。真实仪器上采用哪个安全探测控制、幅度和恢复步骤，必须由仪器负责人依据实际控制接口与标定确定；这里的默认值只用于离线模拟。

## 数据生成

在**工作站**建立独立环境并安装 ML 依赖：

```bash
python3 -m venv .venv-ml
.venv-ml/bin/python -m pip install -r requirements-ml.txt
```

生成互不重叠的训练/验证/测试集合：

```bash
PYTHONPATH=src .venv-ml/bin/python tools/generate_ml_dataset.py \
  --output processed/ml/train.npz --samples 10000 --seed 1001 --input-mode probe-pair
PYTHONPATH=src .venv-ml/bin/python tools/generate_ml_dataset.py \
  --output processed/ml/validation.npz --samples 2000 --seed 2001 --input-mode probe-pair
PYTHONPATH=src .venv-ml/bin/python tools/generate_ml_dataset.py \
  --output processed/ml/test.npz --samples 2000 --seed 3001 --input-mode probe-pair
```

NPZ 保存 `uint8` 图像、20 维初始系数、目标补偿量、固定系数顺序、前向模型版本、预处理版本、场景参数、探测设置和可辨识性声明。过度视野截断的样本会重采样。训练图像采用 99.8 百分位归一化、平方根显示映射，并转成屏幕方向（顶行对应正 `y`）；真实数据必须使用同一 ROI、方向和映射。推理脚本接收已按此约定生成的灰度图，不再做第二次对比度拉伸。

## 训练与离线推理

```bash
PYTHONPATH=src .venv-ml/bin/python tools/train_aberration_network.py \
  --train processed/ml/train.npz \
  --validation processed/ml/validation.npz \
  --output processed/ml/alexnet-regressor.pt

PYTHONPATH=src .venv-ml/bin/python tools/predict_aberration_correction.py \
  --checkpoint processed/ml/alexnet-regressor.pt \
  --image initial.png --probed-image after-known-probe.png
```

训练脚本在具备对应运行环境的机器上优先使用 PyTorch MPS，随后是 CUDA，最后回退 CPU。输出是有效系数补偿 JSON，不会生成或执行仪器命令。模型检查点使用 PyTorch 的受限 `weights_only` 加载路径；仍应只使用自己生成或可信来源的权重。

## 从有效系数映射到 Nion 控制组合

在目标仪器上完成操作员审核的局部响应标定后，把它保存为 JSON：

- `schema_version` 固定为 `nion-response-matrix-1`；
- `terms` 必须是数据集元数据中的完整 20 项规范顺序；
- `actuators` 逐项记录经核对的控制名、单位和 `max_abs_step`；
- `response_mev_per_unit` 是 `20 × 执行器数` 的矩阵，每列表示该控制增加一个单位引起的有效系数变化；
- 可选 `coefficient_weights` 为 20 项正权重，可选 `ridge` 为非负正则强度。

然后在离线 shadow mode 计算单步建议：

```bash
PYTHONPATH=src python3 tools/map_coefficients_to_controls.py \
  --prediction predicted-correction.json \
  --calibration reviewed-response-matrix.json \
  --output control-suggestion.json
```

工具按每个控制的 `max_abs_step` 归一化后做加权岭回归并限幅，输出控制增量、达到限幅的控制、响应矩阵预计实现的 `ΔE` 和尚未校正的残差。它不会读取控制当前值，不输出绝对设定值，也不会连接或调用 Nion Swift。限幅后的建议仍需操作员检查；不能把生成 JSON 当作可直接执行的命令文件。

## 从模拟走向 Nion 的最低验证路线

1. 固定探针会聚角、入口孔径、能量色散、相机长度、ROI、曝光和 detector orientation；保存这些元数据。
2. 用操作员批准的小扰动逐项测量局部响应矩阵 `ΔE = J Δc`。网络先预测 `ΔE`，再用仓库的带阻尼/限幅离线映射求 `Δc`，不要把 `Eij` 当成硬件旋钮值。
3. 采集覆盖多个日期、重新对中和不同信噪比的真实数据，用 session 隔离划分训练/验证/测试，避免相邻帧泄漏。
4. 先做 shadow mode：只显示建议和不确定度，由操作员手动应用；每次只走一小步并重新成像，比较校正前后的 FWHM、RMS、长尾和视野截断。
5. 只有在独立会话数据上验证控制方向、量程、回滚和 interlock 后，再单独评审 Nion Swift 插件。当前仓库不包含自动控制。

不建议一开始直接训练 `图像 → Δc`：设备漂移、维护后的重调和控制耦合会让标签快速失效。把图像反演与本机响应矩阵分开，才能独立检查网络误差、标定误差和执行器限幅。获得真实工作站后，最先需要补充的是：Nion Swift 导出的原始二维阵列样例、当时的采集元数据、可调控制清单及单位、每个控制的小步正负扰动记录，以及恢复到基线的日志。

## 参考工作与源码

- Y. Guo 与 A. R. Lupini, *Automatic and Quantitative Measurement of Spectrometer Aberrations*, Microscopy and Microanalysis 29 (2023), 1671–1681。论文用 Nion 约定定义 `Eij`，说明了单个图样反演可能不唯一，并建议高阶项分阶段测量/校正。
- 作者公开源码：<https://github.com/DrYGuo/EELS-aberration-measurements-and-simulations>。其中包含 Nion Swift 采集代码、波动光学模拟 notebook 和基于 tilt series 的线性回归测量；不是照片中 Florian Heller 的 AlexNet 源码。
- Nion Swift：<https://github.com/nion-software/nionswift>。仪器控制扩展是后续独立阶段，不应与离线训练脚本混合。

截至 2026-09-22，未检索到照片所示 Florian Heller BSc thesis 的公开论文或源码，因此本文只把照片中可见的网络结构作为架构参考，没有声称复现其训练数据、损失、标定或 Nion 控制实现。
