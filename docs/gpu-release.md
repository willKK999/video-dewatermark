# GPU 发布与验收

GPU 镜像是候选配置，当前环境没有 NVIDIA GPU，尚未验证 SAM/ProPainter 的兼容性、显存与吞吐。`telea-dev` 只用于工程冒烟，不计入正式效果验收。

## 模型准备

在可信构建机取得并审查官方源码，固定完整 Git commit：

- SAM 2：[facebookresearch/sam2](https://github.com/facebookresearch/sam2)，SAM 2.1 Hiera Small，配置 `configs/sam2.1/sam2.1_hiera_s.yaml`。
- ProPainter：[sczhou/ProPainter](https://github.com/sczhou/ProPainter)，官方 Release 的 `ProPainter.pth`、`raft-things.pth`、`recurrent_flow_completion.pth`，放在仓库的 `weights/`。
- 可选 YOLO：仅使用授权数据训练且类别名包含 `watermark` 的专用权重。通用 COCO 权重会拒绝运行。启用时在检测环境安装并锁定 ultralytics 版本，同时将其源码/依赖纳入发布记录。

遵循各模型代码、权重与数据许可。文件下载源、HF revision、SHA-256 需与可信发布材料核对，然后生成锁文件。运行期间禁止临时下载。ProPainter 上游会在缺权重时尝试下载，Worker 会在启动前检查三份权重，拒绝不完整缓存。

约定挂载目录：

```
/models/sam2/                   # 完整 Git checkout（含 .git，干净的已提交源码）
/models/sam2/checkpoints/sam2.1_hiera_small.pt
/models/ProPainter/             # 完整 Git checkout
/models/ProPainter/weights/{ProPainter,raft-things,recurrent_flow_completion}.pth
/models/model-lock.json
```

在实际 `/models` 挂载路径下运行（锁文件记录绝对路径）：

```bash
python scripts/lock_models.py \
  --repo /models/sam2 --repo /models/ProPainter \
  --file /models/sam2/checkpoints/sam2.1_hiera_small.pt \
  --file /models/ProPainter/weights/ProPainter.pth \
  --file /models/ProPainter/weights/raft-things.pth \
  --file /models/ProPainter/weights/recurrent_flow_completion.pth \
  --source '记录实际官方 release URL、HF revision 与校验来源' \
  --output /models/model-lock.json
```

`lock_models.py` 计算本地哈希，并不代替来源真实性审核。正式发布还应锁定传递依赖和镜像摘要。GPU requirements 为待兼容性验证的候选 pins，不是已验收的环境锁。

## 构建、运行

```bash
docker build --secret id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt \
  -f deploy/Dockerfile.worker-gpu -t watermark-worker:mvp-1 .
```

SAM 通过 `PYTHONPATH=/models/sam2` 导入官方实现。当前适配器不依赖 SAM 的可选 CUDA 连通域扩展；如具体锁定版本强制要求扩展，在发布镜像构建阶段按官方说明编译，nvcc 与 wheel 均使用 CUDA 12.4，并保留补丁与构建记录。旧 ProPainter/torchvision 导入兼容问题需要在独立环境修补，提交补丁后重新锁源。

GPU Worker 只连接可信控制面 PostgreSQL、Redis 和私有对象桶。Vast.ai 节点不承载 API 或数据库，不开放这些服务的公网端口。使用专用最小权限数据库用户、Redis ACL、S3 凭证和 TLS 连接字符串；通过可信私网/VPN 接入。下面的 `gpu.env` 必须由部署方填入真实地址与密钥，不能沿用 `.env.example` 的本地凭证。

```bash
docker run --rm --gpus 'device=0' --env-file gpu.env \
  -v /absolute/models:/models:ro -v watermark-gpu-work:/work \
  watermark-worker:mvp-1
```

`/work` 需要给 UID 10001 写权限。每张 GPU 只启动一个 Worker；同主机重复进程通过共享 `/work/gpu.lock` 排他。显式配置 `IMAGE_DIGEST`、`PIPELINE_VERSION=mvp-1`、检测模板 `TEMPLATE_PATH`（挂载只读图片）或 `DETECTOR=yolo`/`YOLO_WEIGHTS`，也可在每个镜头提供人工框提示。模板必须与归一后视频中的水印像素尺寸匹配。未配置检测来源且无提示的任务会进入复核。

FP16/BF16 默认关闭。经官方样例、双镜头和音轨测试后才开启 `PROPAINTER_FP16=true`、`SAM_BF16=true`。OOM 首次重排后缩窗到 36 帧；调用方允许降级才使用 540p。再次 OOM 直接失败。总领取最多三次。

## 验收数据与基准

测试集文件格式：

```json
{"samples":[{"id":"sample-01","path":"/data/input.mp4","sha256":"实际SHA256","supported":true,"options":{"max_short_edge":720,"prompts":[{"frame":0,"box":[30,40,120,80]}]},"truth_masks":"/data/masks/sample-01"}]}
```

先冻结 30 条样本（20 支持、10 异常/高难，至少 10 条有合成 Mask 真值），将调参集与最终评测集分开。Mask 真值与归一帧对齐，命名 `000000.png` 起。

```bash
python -m scripts.benchmark --dataset /data/frozen.json --freeze-acceptance --repeats 1 --output acceptance.json
# 单独的 720p/30fps/10秒性能样本，预热后至少20次：
python -m scripts.benchmark --dataset /data/performance.json --repeats 20 --output performance.json
```

脚本校验样本哈希，记录预热与各次执行、分阶段耗时、RTF、nvidia-smi 显存峰值、Mask precision/recall 和失败分母，分别汇总 720p、低分辨率和失败组。PyTorch allocated/reserved 由阶段子进程输出并进入 attempts.metrics。输出帧数、音轨、时长和 ≤80ms 音画偏移在每次执行中检查。

人工逐条记录残留、闪烁、误伤与重叠接缝。有干净母版时可用 `scripts/quality.py` 计算 Mask 内 PSNR/SSIM；这些指标不替代目检。计划门槛为支持集至少18/20成功且目检通过、Mask召回≥90%/精确≥80%、显存≤22GiB、P50≤180s/P95≤300s。尚无实际 GPU 数字。

回滚时先停止新版本领取；只启动与任务 `pipeline_version` 匹配的旧 Worker。运行任务由租约收尾或回收，旧 token 无法更新新尝试。队列最大等待、Worker 心跳、OOM/失败率告警需要部署方接入监控系统；当前数据库 attempts 和结构化日志提供原始数据。
