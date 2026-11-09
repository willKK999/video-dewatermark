# 本次开发验证记录

2026-11-09，在无 NVIDIA GPU 的云开发环境执行。

| 检查 | 结果 |
| --- | --- |
| Go `test ./...`（3项单元测试） | 通过 |
| Go `vet ./...` | 通过 |
| Python 全套测试（9项媒体/算法测试 + 7项集成测试） | 16通过 |
| Python 最终媒体测试重跑 | 9通过 |
| Ruff 错误/未使用导入检查、Python compileall | 通过 |
| OpenAPI 3.0 schema 校验 | 通过 |
| Docker Compose 配置校验 | 通过 |
| Go API 二进制构建 | 通过 |
| 最终异步冒烟 | SUCCEEDED，成功下载MP4 |

最终冒烟使用自有合成视频，320×180、12fps、12帧、1秒，含AAC音轨，`mvp-1-dev` / `telea-dev`。运行真实Go API、PostgreSQL 16.6和Redis 7.4.1；对象存储使用Moto 5.0.23模拟S3。任务ID为 `job_898b7a3808ede31fcee23ef564493fe0`，本地输出 `work/smoke-result.mp4`（不入Git）。

集成测试包括8个并发请求的幂等合并、同键异体409、跨租户404、预签名原对象覆盖后冻结输入保持不变、重复取消、取消与成功竞争、重复消息只领取一次、旧token拒绝、过期租约重领、OOM仅一次低显存重试、配额429及幂等重放不占新配额。

测试发现的终态提交空参数类型推断问题已经修复，并通过重试、取消和竞态测试验证。botocore 在Python 3.12下发出一条 `datetime.utcnow` 弃用警告，不影响测试结果。

尚未验收：真实S3/R2/MinIO签名及权限（MinIO镜像拉取被registry拒绝）、GPU Docker镜像构建、SAM 2.1/ProPainter实际推理、4090显存与吞吐、30条冻结数据集画质。候选依赖和适配代码不能代替这些验收。本次1秒CPU样例不代表方案中的720p/10秒性能指标。

## 合并复核（2026-11-09）

在同一无 GPU 环境把并行实现收进当前仓库后重新执行。没有下载模型权重，没有重建 GPU 镜像。本机没有 Docker，因此没有跑 Compose；存储仍用 Moto 5.0.23，数据库是本机 PostgreSQL 16.15，Redis 7.0.15。

| 检查 | 结果 |
| --- | --- |
| `go test ./...` | 通过（含迁移文件排序） |
| `go vet ./...`、`go build ./cmd/api` | 通过 |
| `pytest worker/tests`（未设置 `INTEGRATION`） | 14 通过，7 项集成测试跳过 |
| `INTEGRATION=1 pytest worker/tests/test_integration.py` | 7 通过。API 已执行 `001`+`002`，覆盖幂等、冻结输入、取消竞争、过期租约、OOM 一次重试和配额 |
| `sha256sum -c docs/spec-v1.1.docx.sha256` | 通过 |
| `002_consistency.sql` 连续执行两次 | 通过；终态回写、无租约的 RUNNING、非成功状态的结果指针、错误内容类型、跨租户上传引用均被拒绝 |
| `worker.store` 领取、成功提交、OOM 重试、取消后提交、过期租约再领取 | 通过；`RUNNING` 直接改为 `CANCELED` 被触发器拒绝 |
| `scripts.smoke` + `telea-dev` Worker | `SUCCEEDED`，任务 `job_146a37475b3107ee3a0621cc9f035eb4`，下载到 320×180、12 帧、H.264 + AAC 的 MP4 |
| `INPAINT_ENGINE=propainter` 启动 | 退出，`ProPainter.pth weight digest unresolved` |
| `telea-dev` 搭配 `mvp-1` | 退出，要求流水线版本带 `-dev` |

这次冒烟只说明合并后的 CPU 工程通路仍能出片。它不是 720p/10 秒性能指标，也不是 SAM/ProPainter 画质验收。
