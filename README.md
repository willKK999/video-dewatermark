# 动态移动视频去水印系统

依据《动态移动视频去水印系统—技术方案 V1.1》实现的首版工程。提供 Go 控制面、PostgreSQL 状态/租约/事务发件箱、Redis Streams 分发、私有 S3 存储，以及分阶段 Python Worker。正式模型链路为检测 → SAM 2.1 Mask → ProPainter → FFmpeg；模型在独立 Python 进程运行，阶段结束释放显存。

当前已验证 CPU 工程闭环，具体结果见 [验证记录](docs/validation.md)。SAM/ProPainter 适配代码和 GPU 候选镜像已提供，实际权重、4090 兼容性、画质与性能仍需按照 [GPU 发布与验收](docs/gpu-release.md) 实测。本仓库不宣称达到方案中的计划性能门槛。

## 本地运行

需要 Go 1.24+、Python 3.10+、FFmpeg/ffprobe（libx264）、Docker Compose。

```bash
cp .env.example .env
python3 -m venv .venv
.venv/bin/pip install -r worker/requirements.txt -r worker/requirements-dev.txt
docker compose up -d postgres redis minio
set -a
. ./.env
set +a
.venv/bin/python -m scripts.init_storage
go run ./cmd/api
```

另外两个终端也加载 `.env`，分别执行：

```bash
.venv/bin/python -m worker.main
.venv/bin/python -m scripts.smoke
```

冒烟脚本生成自有合成视频，申请上传、PUT、创建任务、轮询并下载出片。`.env.example` 默认使用 `mvp-1-dev` / `telea-dev`，它用人工框插值和 OpenCV Telea 验证工程通路。正式配置默认 `propainter`，CPU 冒烟算法不会自动替代正式模型。

也可执行 `docker compose --profile app --profile dev up --build -d`，先创建私有桶；桶初始化脚本从宿主运行。`S3_PUBLIC_ENDPOINT` 必须是客户端可达地址且与签名主机名一致；Compose 内服务访问 `minio:9000`，浏览器/客户端签名地址默认 `localhost:9000`。

### 无法拉取 MinIO 镜像时

本次开发环境的 MinIO registry 拉取被拒绝，存储集成使用 Moto S3 模拟服务验证。它不提供生产 S3 签名校验保障；上线前需用真实 S3/R2/MinIO 重跑验收。本地可按以下方式复现本次测试：

```bash
docker compose up -d postgres redis
.venv/bin/pip install 'moto[server]==5.0.23'
.venv/bin/moto_server -H 127.0.0.1 -p 9000
```

保持该进程运行，再按前述步骤初始化桶、启动 API 和 Worker。生产桶创建与生命周期操作仅由运维凭证执行，Worker 不需要修改桶配置权限。

## API 使用

完整契约见 [OpenAPI](docs/openapi.yaml)。所有 `/v1` 请求携带 `Authorization: Bearer <token>`。`API_TOKENS` 是从部署机密配置注入的 token → tenant 映射，API 启动时校验租户标识。返回时间为 UTC ISO 8601。

1. `POST /v1/uploads`：`filename`、`size_bytes`、`content_type`，返回上传地址与必须原样发送的 `required_headers`（包括签名 Content-Type 和对象标签）。
2. PUT 完成后 `POST /v1/jobs`，必须带 `Idempotency-Key`。同租户同键同体返回原任务，不同体返回409。创建前 HEAD 校验大小/类型，ETag 条件复制至 Worker 专用输入键，避免客户端覆盖已排队素材。
3. `GET /v1/jobs/{id}` 轮询，建议3秒起逐步退避。成功后才签发短期下载地址；到期/丢失产物返回 `result.expired=true`。
4. `POST /v1/jobs/{id}/cancel` 取消排队或运行任务；已成功/失败/待复核返回409，重复取消返回原取消状态。

```json
{
  "upload_id": "上一步返回的upload_id",
  "options": {
    "max_short_edge": 720,
    "allow_downgrade": false,
    "prompts": [{"frame": 0, "box": [30, 40, 120, 80]}]
  }
}
```

`prompts` 是补充的人工提示入口：框坐标以**归一后视频像素**计，帧号从0开始。切镜后需重新提示，或配置检测模板/专用 YOLO。未知水印没有可靠检测结果时进入 `NEEDS_REVIEW`；操作员补框后使用新幂等键创建新任务，不改写旧终态。

## 处理与一致性

- 输入仅支持 MP4/MOV、单路 H.264 8-bit SDR 4:2:0 视频、至多一条主音轨、最长30秒、最大200MiB。HDR、多视频/多音轨和非方形像素拒绝。
- 长边≤1280、短边≤720，等比缩放、保留横竖屏、不放大，转CFR且≤30fps，归一后≤900帧。清单记录时间轴映射、逐帧索引、镜头、框、置信度和Mask哈希。
- 已知模板每5帧匹配或专用YOLO采样检测，镜头独立提示SAM；无命中的镜头密集重检，Mask漂移时用密集检测提示重启跟踪一次。Mask膨胀2像素，面积>20%、剧烈面积变化、明显跟踪偏离或跨窗冲突会进入复核。这些阈值是保守起测值，需在调参集验证。
- ProPainter默认60帧窗口/12帧重叠，串行修复；重叠区域加权拼接，只在Mask内合成。输出MP4/H.264/yuv420p，保留归一主音轨并检查帧数、可解码性和≤80ms音画偏移。
- PostgreSQL条件领取并分配随机token；续租、阶段更新和最终提交均验证token、状态和有效期。Redis重复消息不会重复领取，消息丢失由outbox重投，过期租约由API后台回收。
- 每租户默认最多10个待完成任务，可用 `MAX_PENDING_JOBS` 调整；超限返回429并建议3秒后重试。幂等重放不额外占用配额。
- 同步异常/存储故障指数退避重排，总领取最多3次；首次OOM缩窗并按显式授权降至540p，再次OOM失败。无可靠检测/Mask/接缝则复核，非法媒体失败，不发布结果。
- 取消与成功按数据库行锁竞争，旧尝试不能更新新token的结果。阶段进程每100ms检查取消信号；心跳线程每秒查状态并续60秒租约，取消/失租终止整个进程组。
- 私有桶原片、中间产物按标签保留1天，结果7天；本地隔离目录尝试结束后清理。只有被有效数据库指针引用的成功结果可签名下载。对象生命周期按存储服务的日级调度执行，不保证精确到秒。

## 验证与目录

```bash
go test ./...
.venv/bin/python -m pytest worker/tests/test_pipeline.py -q
# 启动API和基础服务，暂不启动Worker。测试配置需额外加入：
# API_TOKENS 中 "second-tenant-development-token":"second"
INTEGRATION=1 .venv/bin/python -m pytest worker/tests -q
```

集成测试覆盖并发幂等、跨租户隔离、输入不可变副本、取消与提交竞争、过期租约回收、旧token拒绝、OOM只重试一次。随后启动 Worker 运行 `scripts.smoke` 验证真实异步出片。

| 路径 | 职责 |
| --- | --- |
| `cmd/api`、`internal/control` | Go API、分发器、租约回收 |
| `migrations/001_init.sql` | uploads/jobs/outbox/attempts |
| `worker` | 消费、续租、媒体校验、检测、SAM/ProPainter适配、合成 |
| `scripts` | 私有桶初始化、模型锁定、冒烟、基准、PSNR/SSIM |
| `deploy`、`compose.yaml` | 本地部署与GPU候选构建配置 |
| `docs` | OpenAPI、GPU验收与发布步骤 |

待实际部署完成的部分：模型与镜像最终锁定、4090/30条素材验收、GPU依赖兼容性修补、真实S3签名与权限验证、监控告警接入、复杂跳位与半透明水印的检测参数调优。阶段二 DiffuEraser 与阶段三自动引擎路由按方案留待后续，不计入首版闭环。
