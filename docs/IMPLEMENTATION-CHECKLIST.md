# V1.1 实现与复核记录

本文是共享云电脑上那套并行源码（当时路径 `/workspace/video-dewatermark`，入口 `cmd/server`，没有 git 提交）的复核记录。它不是本 GitHub 仓库合并之后的测试报告。合并后的单一代码路径以 `cmd/api` 和 `telea-dev` 为底座，验证数字见 [validation.md](validation.md)。

下面这些约束已经接进当前仓库，而不是再放一套并列工程：终态行不可改写、生产权重 digest 未公布则启动失败、CPU 冒烟引擎不能冒充正式流水线版本、结果对象按尝试隔离并在提交前核对。清单里的 Go 10 项和 Python unittest 24 项是那棵源码树上的结果。

## 已交付

- [x] 完整阅读原 SPEC，先写 IMPLEMENTATION-PLAN，再写失败测试。初始 Go/Python red 日志保存在忽略的 `.codex-out/`；随后实现并变绿。
- [x] Go `/v1` Bearer → tenant；上传签名、HEAD + ETag 条件 COPY 不可变 input；job 幂等、租户隔离、轮询、取消、统一错误与 UTC 时间戳。
- [x] PostgreSQL uploads/jobs/outbox/attempts；组合租户 FK/幂等唯一约束、状态/尝试约束、终态 immutable trigger。每租户 advisory lock 串行化活跃任务配额与幂等检查。
- [x] 同事务创建/重排 + outbox；Redis 仅 job_id；XADD 后 sent；重复消息安全；过期租约 reaper、取消收尾、滞留 queued 补投。
- [x] Python 真实 PostgreSQL/Redis/S3 adapters，条件 claim/递增 attempt/随机 token；所有 heartbeat/stage/success/fail 均 fenced；总计最多 3 claims；一次 OOM 低显存重试；指数退避 + jitter；pending 回收保存扫描 cursor。
- [x] 尝试隔离目录和结果 key；下载/阶段/子进程/传输检查失租；结果验证后上传并 HEAD，再条件提交；取消或旧尝试不能覆盖有效指针；清理临时目录。
- [x] FFprobe/FFmpeg H264 8bit SDR MP4/MOV、长度/大小/帧数限制、旋转、非零 start、VFR→CFR≤30fps、SAR/比例与横竖屏、无放大、音轨 AAC/复制；FFV1 lossless 中间编码；发布前全解码/帧数/尺寸/时长和音视频时间戳≤80ms 检查。
- [x] OpenCV 已知模板移动检测；每 5 帧提示 + 密集检测；切镜/跳位重置、消失/漂移 NEEDS_REVIEW；YOLO 本地模型插件接口。
- [x] SAM2.1 Small 视频预测器 wrapper、Tiny 低显存切换；逐帧 0/255 Mask、索引/尺寸/面积/漂移限制；跟踪模型释放后修复。
- [x] ProPainter 官方推理 subprocess/offline 资产接口；主模型+RAFT+flow completion；60/12 窗口、重叠 Mask 内混合和接缝指标；显式 mock 仅 CPU 测试。DiffuEraser 仅 phase2 stub。
- [x] 结构化允许字段日志、队列/阶段/分辨率/重试/版本/显存指标；不打印媒体/令牌/签名 URL/原始 subprocess stderr。
- [x] Compose + API/CPU/GPU Dockerfile、私有 MinIO bootstrap/lifecycle、中文 README、精确 API/local 环境示例、后续 4090 准备说明、合成授权 fixture/smoke 脚本。
- [x] 官方模型 manifest：SAM commit/HF revision/SHA256 已由官方 metadata 获得；ProPainter 官方 metadata digest=null 明确 unresolved，生产 fail closed。未下载 weights/datasets。
- [x] `.gitignore` 覆盖 .codex-out/.venv/cache/credentials/generated media/weights；原 DOCX SHA256 未变。

“已交付 wrapper”表示代码实现，不表示真实模型推理或画质通过。

## 实际执行与结果

开发机：Go `1.27.3 linux/amd64`，Python `3.13.5`，FFmpeg/ffprobe `7.1.5-0+deb13u1`；CPU venv 按 requirements 安装。项目 Docker 目标是 Python3.10.19 与 Go1.23.6；这些镜像没有在本机构建。

无 Docker。为覆盖真实 SQL，从 Debian 官方包仅提取 PostgreSQL `17.11-0+deb13u1` 服务二进制到忽略的 `.codex-out/pg-dist/`，用非 root 用户启动独立临时 cluster，监听 `127.0.0.1:55432`。没有接触其他项目或现有数据库。每项集成测试建立随机 schema，完成后只删除其自身 schema。

实际初始化命令：

```bash
.codex-out/pg-dist/usr/lib/postgresql/17/bin/initdb -D .codex-out/pgdata --no-locale --auth=trust
.codex-out/pg-dist/usr/lib/postgresql/17/bin/pg_ctl -D .codex-out/pgdata -l .codex-out/postgres.log -o '-p 55432 -h 127.0.0.1 -k /workspace/video-dewatermark/.codex-out' start
```

最终执行命令：

```bash
PGPASSFILE=/dev/null PGSERVICEFILE=/dev/null TEST_DATABASE_URL=postgres://box@127.0.0.1:55432/postgres go test ./... -count=1 -v
PGPASSFILE=/dev/null PGSERVICEFILE=/dev/null TEST_DATABASE_URL=postgres://box@127.0.0.1:55432/postgres .venv/bin/python -m unittest discover -s worker/tests -v
go build ./...
go vet ./...
.venv/bin/ruff check worker scripts --select F
sha256sum -c docs/spec-v1.1.docx.sha256
```

结果：Go 10 项测试 PASS（含真实 PostgreSQL 的并发幂等、原子 outbox、租户隔离、quota、reaper/stranded、终态约束；Redis publisher 使用 miniredis；S3 adapter 使用 HTTP 协议 fake）。Python 24 项全部 OK，无 skip（含 6 项真实 PostgreSQL adapter 测试；FFmpeg 真实生成/归一/输出验证；mask 索引；取消/失租；worker fake claim/submit/cleanup；OOM 分类/降级授权；pending cursor）。Go build/vet 和 Ruff F 检查 exit 0。DOCX hash check OK：`e8ae5859af3cb16a9f1de5730d8752a25c5def092dd20073c1abb746ad5000d3`。

补充执行：`go test -race ./...`（同一 TEST_DATABASE_URL）；自制 60 帧移动模板通过真实 OpenCV 检测 + 显式 mock tracker/inpainter + FFmpeg 验证的 Worker 编排检查。最终结果见本文件末尾复核结论。

## 复核重点与修正

- 检查所有 API 资源访问均带 tenant，worker/publisher/reaper 是可信服务角色；SQL 参数绑定，HTTP 不暴露原始 DB/存储错误。
- 验证并发幂等只创建一个 job/outbox；配额不能阻止已有幂等任务返回；上传复制的丢失竞态只产生受生命周期回收的孤儿 input。
- 校验状态、租约、attempt 与 OOM 上限；旧 token/过期 lease/cancel 后更新拒绝；终态不能通过 SQL 回滚状态。
- 修正 PostgreSQL SQL_ASCII 测试环境返回 bytes 的问题：Python 显式 UTF-8；生产连接有 connect/statement/lock timeout。
- 修正 FFmpeg7 测试旋转 metadata 的生成方式；覆盖真正 Display Matrix、非零起点、VFR 与非方像素。共同视频起点对齐音频，保留主音轨并检查首尾差。
- 修正 Redis pending 回收 cursor，防止仅扫描前缀造成饥饿；ProPainter 自动下载函数替换为 local-only；source mounts 的 Git 安全目录仅对单次命令设置，不改全局 Git 配置。
- 补入发布阶段持久化计时、lossless 中间编码、只合成 Mask 内和原始 DOCX hash。

## 仍未验证 / 后续必做

- [ ] 没有 Docker，Compose/Dockerfile 构建、真实 Redis pending 与 MinIO 私有 PUT/COPY/GET/生命周期全栈 smoke **未执行**。交付 `scripts/smoke.py`；必须在有 Docker 的机器执行。当前 S3/Redis 协议测试不是这些服务的替代验收。
- [ ] SAM/ProPainter weights 未下载/运行，CUDA/4090/PyTorch3.10 容器组合和 Tiny/OOM 实机行为未验证；模型 adapter 仍可能需针对上游依赖调整。ProPainter 三项 checksum unresolved 是生产硬阻断。
- [ ] 镜像平台 digest、完整 GPU transitive lock、最小权限生产 IAM、TLS ingress、真实 S3/R2 账户互通及告警平台待上线前完成。当前固定 image version tags 和直接 Python 包版本不等于全供应链 digest 锁定。
- [ ] HDR/复杂多轨、YOLO 训练、人工提示 UI、30 条冻结授权素材评测、真实 Mask 精度/观感/PSNR/SSIM、RTF/P50/P95/显存基准均未完成。所有 SPEC 建议门槛是 UNMEASURED PLAN。
- [ ] 检测/面积/漂移/接缝阈值为起测参数，无法保证任意移动/半透明水印；未知模板或不安全结果进入 NEEDS_REVIEW。
- [ ] 全片帧/Mask 暂存内存，900 帧/720p 上限的 RAM/磁盘压力尚未验证；单个 CUDA kernel 的即时抢占不承诺。内容级 AV 对齐仍需授权人工样片检查。
- [ ] DiffuEraser、引擎路由、跨版本中间产物复用、运营 dashboard/告警与生产发布均延后。

未执行远端 push、创建 origin/GitHub、云端代理委派、付费/部署或破坏性 Git 操作。

## 最终复核结论

`go test -race ./...` 在同一临时 PostgreSQL 上 PASS；60 帧自制移动模板 CPU 编排返回 SUCCEEDED（仅 OpenCV 检测和 FFmpeg 为真实媒体处理，tracker/inpainter 明确 mock）。`git diff --cached --check` 和 `git diff --check` 均无错误；新增文件已暂存以便完整审阅，没有创建 commit 或远端。复核后再次执行全部 Go/Python CPU 测试和 Go build，均通过。临时 PostgreSQL 在测试结束后正常停止，其数据仅留在忽略的 `.codex-out/`。
