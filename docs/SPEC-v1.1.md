视频去水印系统—技术方案

V1.1  ·  2026 年 11 月 9 日  ·  技术选型与实施设计

首周在 Vast.ai 单张 RTX 4090 24 GB 上验证授权短视频的异步处理闭环。用 OpenCV 或专用 YOLOv8n 检测移动水印，SAM 2.1 生成逐帧 Mask，ProPainter 进行时序修复，FFmpeg 合成视频。处理链路以 720p 优先，兼容性、效果和吞吐由基准测试验证。

01  系统模块与实施边界

模块

技术与职责

控制面

Go API 处理鉴权、上传签名、任务创建、查询和取消。PostgreSQL 保存状态、租约和事务发件箱，Redis Streams 负责分发。

GPU 执行

PyTorch Python Worker 领取任务，管理下载、逐阶段执行、指标、清理与结果提交。检测与跟踪、修复可使用独立运行环境，按阶段独占 GPU。

定位与跟踪

OpenCV 处理已知模板及运动线索。YOLOv8n 使用自有或授权数据训练水印类别，SAM 2.1 Hiera Small 用框或人工提示传播逐帧 Mask。

媒体处理

FFmpeg/ffprobe 校验、抽帧、编码和音轨合成。ProPainter 接收对齐的帧和二值 Mask，输出时序修复画面。

存储与部署

私有 S3 或 R2 存放原片、阶段产物及结果，Vast.ai 只运行 Docker GPU 执行面。Go、Redis、PostgreSQL 部署在可信控制面。

范围与后续阶段

初版支持 MP4 或 MOV 容器内的 H.264、8-bit SDR，输入最长 30 秒、最大 200 MB、归一后最多 900 帧。按长边 ≤1280、短边 ≤720 等比缩放，保持横竖方向和原始比例，默认不放大。对 VFR 固定帧率后统一建索引，默认至多 30 fps。HDR 与复杂多轨素材延后。

阶段一采用 Docker 与可版本化脚本，无 ComfyUI 依赖。阶段二在同一 Mask 和同一评测集上评估 DiffuEraser。阶段三加入可替换的修复引擎，按 Mask 面积、运动难度及实测画质和时延路由。

02  部署依赖与显存策略

层级

候选基线与验证项

GPU 宿主

Linux x86_64、Docker、NVIDIA 驱动与容器 GPU 运行支持。选一张 4090 24 GB、可验证驱动与网络的主机。起测配置为 8 vCPU、32 GB RAM、80 GB 可用盘，按素材峰值调整。

跟踪环境

Python 3.10、PyTorch 2.5.1 / torchvision 0.20.1、CUDA 12.4 wheel 作为起测组合。SAM 2 官方最低版本见来源；编译扩展时令 nvcc 与 wheel 对齐。

修复环境

先独立锁定 Python 3.10 与上述 PyTorch/CUDA 组合并运行 ProPainter 样例，遇旧依赖冲突时在独立容器修补与验证。上游旧版最低依赖不能作为 4090 兼容证明。

控制与媒体

Go、Redis Streams、PostgreSQL、opencv-python-headless、FFmpeg/ffprobe 使用镜像摘要和锁文件固定。启动时核查 FFmpeg 编解码器、GPU、权重校验与对象存储读写。

模型制品

固定 Git commit、HF revision、文件 SHA-256及下载源。构建期预下载，推理时禁用临时拉取。缓存挂载为只读。

720p 优先与资源上限

单卡并发 1。SAM 2.1 Small 优先，Tiny 用于显存受限样本。跟踪与修复串行，阶段结束释放模型或退出进程。SAM 可测试 BF16 autocast，ProPainter 可测试 FP16，检查 NaN、花屏和 Mask 偏移后再固定配置。以 nvidia-smi 进程峰值和 PyTorch allocated/reserved 同时记录显存。

按镜头及窗口切分，初测从 60 帧窗口、12 帧重叠起步，记录重叠区接缝。窗口内限制全局参考与邻帧，跨窗口用重叠帧锚定位置。显存不足时先缩窗，再减参考或降低处理尺寸。降至 540p 需调用方允许，响应须标明实际分辨率。720p 在 24 GB 上的吞吐和稳定性均待实测。

部署与回滚

先用测试素材跑 CUDA 张量、官方短片、双镜头自制样片及音轨检查。发布记录镜像摘要、模型版本和参数集，作业固定 pipeline_version。回滚时停止新版本领取，旧版本只接兼容任务，运行中的尝试按租约收尾或回收。

03  端到端流程与异常回退

从上传到出片

1. 客户端向 Go API 申请预签名上传地址，直传私有桶。创建任务时 API 验证租户、对象归属及大小，同事务写入任务和发件箱。发布器将 job_id 送入 Redis。

2. Worker 原子领取任务，下载至隔离目录。ffprobe 检查容器、编码、时长和尺寸，FFmpeg 按统一旋转、分辨率、帧率及起点解码。记录原片与归一片时间轴映射。

3. 按镜头切分，以已知模板匹配或水印专用 YOLOv8n 在关键帧检测，初测每 5 帧一检。将高置信框转为 SAM 2.1 提示。遇切镜、跳位、消失重现或框与掩膜偏离时重新检测并初始化。半透明和低对比水印应单独评估。

4. 逐帧生成同尺寸、同索引的 0/255 Mask。按验证结果小幅膨胀以覆盖阴影和抗锯齿边缘，限制面积与突变。修复后只在 Mask 或受控羽化边界内合成，避免全画面漂移。抽检跨窗接缝、闪烁与误伤。

5. 以 MP4 H.264/yuv420p 输出，保留主音轨并按兼容性复制或转 AAC，维持归一后的时间轴。检查可解码、帧数、时长及音画同步，上传尝试专属对象键，再提交成功状态。查询成功结果时生成短期下载链接。

失败判定与重试

触发条件

处理与对外结果

检测无命中或漂移

切镜重置、提高采样密度，仍低置信或疑似误伤则 NEEDS_REVIEW。操作员可补关键帧框，以新任务重跑。

GPU OOM

终止失败进程释放显存，按锁定的低显存配置重试一次。允许降级时可选 540p；仍失败则 GPU_OOM，保留参数和峰值。

主机中断或存储超时

指数退避并加抖动，任务最多 3 次领取，含 OOM 尝试；租约超时后可换主机。仅复用校验通过且版本一致的产物。

媒体非法或出片异常

输入不合规终止为 INVALID_MEDIA。输出坏帧、音画超差或接缝严重时置 OUTPUT_INVALID 或 NEEDS_REVIEW，不发布下载地址。

取消或租约失效

阶段间检查，长进程轮询后终止进程组。旧尝试不能覆盖有效结果，孤儿对象按生命周期清理。

04  API 合约

基础路径 /v1，所有请求使用 Bearer Token，按 tenant_id 隔离。响应时间戳使用 UTC ISO 8601，ID 为不透明字符串。初版采用轮询，排队时建议 3 秒起步并逐步退避。

方法与路径

成功语义

POST /uploads

201。请求 filename、size_bytes、content_type，响应 upload_id、put_url、required_headers、expires_at。客户端按签名要求 PUT。

POST /jobs

202。必填 Idempotency-Key，提交已上传对象和处理选项。相同租户同键同体返回原任务，不同请求体返回 409。

GET /jobs/{id}

200。返回 status、stage、attempt、result、error 与时间戳。成功且对象有效时签发短期下载地址，过期产物返回 result.expired=true。

POST /jobs/{id}/cancel

202。排队或运行任务发起取消，重复请求返回现有取消状态。已成功或已失败等不可取消状态返回 409。

状态与阶段

QUEUED → RUNNING → SUCCEEDED / FAILED / NEEDS_REVIEW。可重试故障回到 QUEUED。QUEUED 或 RUNNING 可进入 CANCEL_REQUESTED → CANCELED。各终态不可改写，人工复核重跑创建新任务。RUNNING 的 stage 可为 probe、detect、mask、inpaint、encode、publish。

创建任务请求与成功查询响应

{  "upload_id": "upl_01",  "options": {"max_short_edge": 720, "allow_downgrade": false}}

{  "id": "job_01", "status": "SUCCEEDED", "stage": null,  "attempt": 1, "pipeline_version": "mvp-1",  "created_at": "2026-11-09T02:00:00Z",  "updated_at": "2026-11-09T02:03:00Z",  "result": {    "download_url": "https://storage.example/signed/result",    "expires_at": "2026-11-09T02:13:00Z", "expired": false,    "width": 1280, "height": 720, "duration_ms": 10000  },  "error": null}

创建成功返回同一结构，初始 status=QUEUED、attempt=0，stage/result/error 均为 null。4xx/5xx 使用 error.code、message、retryable 和 request_id。典型 HTTP 错误为 400 参数、401 鉴权、404 不存在或无权、409 冲突、413 超限、422 媒体或上传校验、429 配额及 503 服务暂不可用。异步失败仍可 GET 200，error 填业务错误。

05  数据结构与一致性

对象

最小字段与约束

uploads

id、tenant_id、object_key、size_bytes、content_type、checksum、expires_at。任务创建前 HEAD 核验并绑定不可变输入（版本 ID 或转存至写入受限键）。

jobs

id、tenant_id、upload_id、idempotency_key、request_hash、status、stage、attempt、lease_token、lease_until、options、pipeline_version、error_code、result_key、created_at、updated_at。唯一键为 tenant_id + idempotency_key。

outbox

id、job_id、dispatch_no、available_at、sent_at。创建或重排任务时同事务落表。发布失败可重试，重复消息由数据库领取条件吸收。

attempts

job_id、attempt_no、worker_id、started_at、ended_at、config_hash、peak_vram_mib、phase_ms、failure_code。以 job_id + attempt_no 为主键。

媒体清单

frame_index 从 0 起，含 pts_ms、shot_id、bbox_xyxy、confidence、mask_key。框和 Mask 使用归一后的像素坐标，Mask 文件按帧号命名。修复阶段须核对帧数、尺寸、版本及校验值。

领取、提交与取消

Redis 只传递作业标识。Worker 用 PostgreSQL 条件更新从 QUEUED 领取并递增 attempt，写入随机 lease_token 与到期时间。更新阶段、续租、终态提交均匹配当前 token 与有效状态。收到重复消息时由数据库状态决定能否领取。

发布器确认 XADD 后标记已发，崩溃时允许重发。Worker 在数据库成功领取或确认消息已失效后 ACK，Redis 消费组回收未确认消息。定时扫过期租约并在同一事务重排或标失败，同时补投滞留 QUEUED 任务，覆盖消息丢失和 Worker 退出。

对象键按租户、任务、尝试划分。结果写入后核验再条件更新数据库指针，查询仅读取有效指针。取消与成功提交按行状态竞争，先写入者生效。无有效指针的产物由桶生命周期删除。默认原片和中间产物保留 24 小时，结果 7 天，按租户策略调整。

可观测性

记录 job_id、attempt、镜像摘要、模型版本、排队与各阶段耗时、显存、重试、实际分辨率和结果码。告警覆盖队列最老等待、连续 OOM、Worker 心跳缺失及失败率。日志中遮蔽签名 URL、令牌和素材内容。

06  技术验收与基准测试

冻结 30 条测试素材，每条 5–30 秒。20 条在初版支持范围内，覆盖横竖屏、移动、半透明、切镜和运动背景，其中至少 10 条用干净母版合成水印并保留真值。另 10 条覆盖无水印、异常输入和高难样本。开发调参与最终评测分开，按样本记录质量与失败分类。

指标

建议门槛（计划值，未实测）

完整链路

支持集 20 条中至少 18 条成功且通过目检。所有已成功结果能解码、帧数与归一输入一致，音画偏移 ≤80 ms。故障用例无重复发布与跨租户访问。

Mask 与观感

合成集按标注帧计算 Mask 像素召回率 ≥90%、精确率 ≥80%。支持集成功样本逐条检查残留、闪烁、误伤和接缝；有母版时另记录 Mask 区域 PSNR/SSIM。

速度与显存

720p/30 fps/10 秒，单卡预热后重复 ≥20 次。起测目标峰值 ≤22 GiB，执行耗时 P50 ≤180 秒、P95 ≤300 秒。排队和冷启动单独统计，记录降级与失败分母。

以执行耗时 ÷ 视频秒数计算实时因子 RTF。汇报检测、跟踪、修复、编码、上传的分段耗时，按 720p、降级、失败分组。未达门槛的样本记录复现材料并保留复核或降级路径。

模型与权重入口

SAM 2.1 采用 Meta 官方视频预测器和 Hiera Small 权重，显存受限时切换 Tiny。 [官方仓库] [HF Small]

ProPainter 从官方 Release 获取主模型、RAFT 和光流补全权重。 [官方 Release]

阶段二按 DiffuEraser 官方推理脚本接入，保留同一套 Mask 与测试集进行对照。 [官方仓库] [HF 权重]

## 官方模型链接

- SAM 2: https://github.com/facebookresearch/sam2
- SAM 2.1 Hiera Small: https://huggingface.co/facebook/sam2.1-hiera-small
- ProPainter v0.1.0: https://github.com/sczhou/ProPainter/releases/tag/v0.1.0
- DiffuEraser (phase 2): https://github.com/lixiaowen-xw/DiffuEraser
- DiffuEraser weights: https://huggingface.co/lixiaowen/diffuEraser/tree/main
