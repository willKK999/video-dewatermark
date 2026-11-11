# 视频去水印系统

依据《视频去水印系统—技术方案 V1.1》实现的首版工程。原文见 [SPEC-v1.1.md](docs/SPEC-v1.1.md)，原 DOCX 与 SHA-256 一并保留。提供 Go 控制面、PostgreSQL 状态/租约/事务发件箱、Redis Streams 分发、私有 S3 存储，以及分阶段 Python Worker。正式模型链路为检测 → SAM 2.1 Mask → ProPainter → FFmpeg；模型在独立 Python 进程运行，阶段结束释放显存。

当前仓库是两套 V1.1 实现合并后的**唯一代码路径**：云端已跑通的 API、`telea-dev` / `mvp-1-dev` 冒烟和 OpenAPI 作为底座；并行实现里的规格、模型来源清单和生产失败闭环融进同一套进程，没有第二套 `cmd/server` 或另一份 Worker。对比结论见下文「合并取舍」。CPU 工程闭环见 [验证记录](docs/validation.md)。SAM/ProPainter 适配代码和 GPU 候选镜像已提供，实际权重、4090 兼容性、画质与性能仍需按照 [GPU 发布与验收](docs/gpu-release.md) 实测。本仓库不宣称达到方案中的计划性能门槛，也没有下载多 GB 权重。

仅处理自有或已获授权的短视频，不处理盗版或 DRM 绕过。

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

若本机 PostgreSQL 数据卷是合并前创建的，请重新建库或让 API 以 `AUTO_MIGRATE=true` 再执行一次。启动时会按文件名顺序应用 `migrations/*.sql`，其中 `002_consistency.sql` 为终态保护、租约约束和尝试记录补列，重复执行是安全的。`dev` Worker 使用 `telea-dev`，不要和正式 `propainter` Worker 同时消费同一 Redis 流。

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

- MP4/MOV 输入支持单路 H.264 8-bit SDR 4:2:0 视频、至多一条主音轨、最长30秒、最大200MiB。HDR、多视频/多音轨和非方形像素拒绝。
- HLS 长视频可提交 m3u8 对象键或链接，自动选择码率、连续分块并携带重叠上下文，输出 HLS 与 MP4、复制原 AAC 音轨。操作台支持 HLS 播放、分块进度、取消和未完成块重试。格式限制、接缝策略及验收见 [HLS 使用说明](docs/hls.md)。
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
.venv/bin/python -m pytest worker/tests -q
# 启动API和基础服务，暂不启动Worker。测试配置需额外加入：
# API_TOKENS 中 "second-tenant-development-token":"second"
INTEGRATION=1 .venv/bin/python -m pytest worker/tests -q
sha256sum -c docs/spec-v1.1.docx.sha256
```

集成测试覆盖并发幂等、跨租户隔离、输入不可变副本、取消与提交竞争、过期租约回收、旧token拒绝、OOM只重试一次。随后启动 Worker 运行 `scripts.smoke` 验证真实异步出片。

| 路径 | 职责 |
| --- | --- |
| `cmd/api`、`internal/control` | Go API、分发器、租约回收 |
| `migrations/` | uploads/jobs/outbox/attempts，以及终态和租约约束 |
| `worker` | 消费、续租、媒体校验、检测、SAM/ProPainter适配、合成 |
| `scripts` | 私有桶初始化、模型锁定、冒烟、基准、PSNR/SSIM |
| `deploy`、`compose.yaml` | 本地部署与GPU候选构建配置 |
| `docs` | 原规格与校验和、OpenAPI、模型来源、GPU验收与验证记录 |
| `deploy/models.json` | 官方模型来源与 SHA-256；ProPainter 未公布 digest 时生产启动失败 |

## 合并取舍

云端版本（本仓库 `29fe540`）保留为运行底座：`cmd/api`、OpenAPI、人工补框 `prompts`、`mvp-1-dev` / `telea-dev` 的 OpenCV Telea 出片、Moto/PostgreSQL/Redis 冒烟、pytest 集成（幂等、冻结输入、取消竞争、过期租约、OOM 一次低显存重试）。错误体保持顶层 `request_id`，与 OpenAPI 一致。Bearer 比较使用常时间比较。

从并行实现融入、并接到这一条路径上的部分：

- 原规格 `docs/SPEC-v1.1.md`、`docs/spec-v1.1.docx` 及其 SHA-256，以及模型来源说明。
- `deploy/models.json`。`INPAINT_ENGINE=propainter` 在消费队列之前检查清单；任一必需权重的 SHA-256 不是 64 位十六进制（当前 ProPainter 三项为 `UNRESOLVED`）则进程退出。`telea-dev` 必须搭配以 `-dev` 结尾的 `pipeline_version`，不能冒充 `mvp-1`。操作员的 `MODEL_MANIFEST` 还须覆盖清单中每个已公布 digest，包括 SAM Tiny。ProPainter 子进程会替换上游的自动下载函数，只读本地文件。
- 数据库终态不可改写、合法状态迁移、运行中必须持有租约、结果指针仅允许出现在成功状态、上传内容类型约束，以及按租户绑定的上传外键。重复取消直接返回当前状态，不再回写终态行。迁移在 `002_consistency.sql`。
- 领取记录写入镜像摘要和模型版本。Redis `XAUTOCLAIM` 记住扫描游标。队列消息的流水线版本与本 Worker 不一致时不 ACK，留给对应版本的 Worker。结果上传后同时核对长度和 SHA-256 元数据。
- 数据库连接使用 UTF-8，并设置语句和锁超时。

有意不并入的部分：并行实现的 `cmd/server`、另一套 Worker 包和 `explicit-cpu-mock` 出片。那条 CPU 通路用矩形 mock 代替修复，不能替代已经冒烟下载出真实 MP4 的 Telea 通路。取消语义保留云端行为：排队任务直接 `CANCELED`，运行中任务进入 `CANCEL_REQUESTED`，由持有租约的 Worker 或租约回收收尾；没有改成「所有取消都先等 reaper」。OOM 次数仍由 `attempts.failure_code=GPU_OOM` 判定，不另加一套 `oom_retries` 计数。配额环境变量仍是 `MAX_PENDING_JOBS`（默认 10）。Compose 仍使用已固定摘要的 PostgreSQL 16.6 / Redis 7.4.1，以及已经验证过的 Moto 替代路径；并行实现的 Compose 未在无 Docker 的机器上跑通，不替换当前文件。

待实际部署完成的部分：填入并核实 ProPainter digest 之后才能启动正式 Worker、模型与镜像最终锁定、4090/30 条素材验收、GPU 依赖兼容性修补、真实 S3 签名与权限验证、监控告警接入、复杂跳位与半透明水印的检测参数调优。阶段二 DiffuEraser 与阶段三自动引擎路由按方案留待后续，不计入首版闭环。并行实现当时的 Go/Python 命令记录留在 [实现清单](docs/IMPLEMENTATION-CHECKLIST.md)，那是另一棵源码树上的结果，不能代替本仓库的 [验证记录](docs/validation.md)。

### 中文网页操作台

Go API 直接托管中文操作台，浏览器打开 API 根路径即可输入访问令牌、上传视频、管理任务、播放对比及下载成品。网页采用同源代理上传，无需配置 R2 浏览器跨域。公网部署、专用令牌文件、HTTPS 证书说明和测试方式见 [操作台说明](deploy/console/操作台说明.md)。

HLS 已支持多水印独立轨迹、时域共识与完整图标复检。黄果本轮按要求只验收第1块，通过后停止，不自动继续整集；实测状态和范围见 [HLS 使用说明](docs/hls.md)。
