# SAM 3.1 文字视频掩膜来源

此来源默认关闭，只在显式配置 `enabled: true` 时运行，不接入补画或默认生产流程。
依赖官方 facebookresearch/sam3 的 SAM 3.1 multiplex 视频接口，权重使用本地文件，禁止运行时自动下载或远程推理。

```sh
python -m worker.sam31_masks --frames /输入帧目录 --output /全新输出目录 --checkpoint /模型/sam3.1_multiplex.pt --config /配置.json --baseline /可选基线掩膜目录
```

配置范例为 `config/sam31-mask.example.json`。按固定长度窗口、每个通用文字提示独立建立视频会话，在窗口首帧输入文字，再逐帧视频传播；后续新实例由官方检测跟踪器发现。窗口限制内存，不根据片子、帧号或坐标作判断。不同窗口的对象ID不混用。置信度过滤后按对象ID的出现帧数及生命周期内出现比例过滤。持续性只能排除瞬时/稀疏候选，无法保证排除持久字幕、招牌或人物，仍须目检。

输出 `masks/000000.png` 等保持输入文件名、尺寸及0/255二值格式；`merged-masks/` 是与基线的严格并集，不能删除基线误报。`raw-object-masks/` 保存原始候选；`layers/` 保存通过持续性和面积保护的对象分层；`source.json` 绑定配置、权重和输入帧SHA256；`metrics.json` 包含轨迹、逐帧像素、各阶段耗时、图层及其逐帧摘要。

单对象超过20%会整条拒绝并记录原因，不能切碎绕过保护；其他对象按最大面积保守装箱，每一层逐帧复查≤20%。联合掩膜可能超过20%，不能作为单层送入补画；合并基线也不能自动视为已经验证的单层，调用方必须保留和复核原基线图层。空结果仍输出完整780帧（或实际输入帧数）零掩膜；缺帧、模型返回不完整或错误二值数据会失败，不宣称成功。

官方代码固定版本：`eef6839bcfc3458dd52ea80535dc6bd62aea86b1`。
参考：https://github.com/facebookresearch/sam3/blob/main/examples/sam3.1_video_predictor_example.ipynb 。
cleanplate只参考其通用文字概念选择，不引入其ComfyUI和补画服务：https://github.com/qilaidev/cleanplate 。

该固定官方版本的共用会话接口会传入 `offload_state_to_cpu=False`，但multiplex模型的 `init_state` 没有这个参数。适配器通过签名检测仅移除这个不支持且关闭的选项；若请求开启则明确拒绝，不假装提供CPU状态卸载。CPU视频帧卸载保持开启。文字在窗口首帧初始化，传播明确只向前，避免官方默认双向路径重复返回首帧。此兼容在运行前静态检查发现并补充行为测试。

4090首次运行在官方16帧检测批量上显存不足，唯一一次推理错误恢复将检测批量及后处理批量统一设为1（`inference_batch_size`）。窗口长度、输入分辨率、对象容量、提示和语义阈值不变；这只限制资源峰值，不按影片改变算法。首次失败日志与输出另存，恢复输出用全新目录，仍失败则停止。
