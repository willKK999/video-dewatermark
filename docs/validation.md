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
