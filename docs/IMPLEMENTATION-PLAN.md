# V1.1 首周实现计划

这是并行实现动工前的计划，保留作规格阅读顺序的记录。当前仓库的实际代码路径以 GitHub 云端版本为底座，见仓库 README 的「合并取舍」。

已完整阅读 SPEC-v1.1.md；原 DOCX 保持不变。

1. 先写控制面合约/状态机和 Python 媒体/掩膜/Worker 测试，确认失败后实现。
2. 实现 Go HTTP、租户鉴权、PostgreSQL 事务/租约/发件箱、Redis Streams、私有 S3 不可变输入。
3. 实现串行 Python Worker、真实队列/数据库/存储适配器、FFmpeg 管线、OpenCV 检测、显式模型接口与 CPU 测试模式。
4. 添加 Compose、GPU 镜像、模型清单与中文操作说明；生产模型缺失时 fail closed。
5. 运行所有 CPU 测试与 Go build，复核竞争条件、日志、git diff，记录未验证项。

不下载 weights/datasets，不部署，不声称 GPU 质量或速度达标。默认 pipeline_version=mvp-1。
