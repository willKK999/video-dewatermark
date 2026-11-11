# 视频去水印系统：工作约定与黄果验收

## 工作与安全

全程使用简体中文说明、写文档及提交信息，代码标识符除外。保留工作区已有修改。本提速工作树固定为 `/workspace/vdw-speedup`，使用分支 `perf/speedup-v2`，起点为 `58dc8df`，提交作者为 `willkk <willkk@users.noreply.github.com>`，只做本地提交，不推送、不创建合并请求。

禁止读取 `key.bin`，禁止复制或上传密钥；禁止打印、提交密钥、令牌及预签名地址。环境配置只加载，不输出。视频、图片验收产物、模型和分片不能提交进仓库。黄果明文素材已经上传，不要重新解密。

## 提速任务隔离（优先执行）

主工作目录 `/workspace/video-dewatermark-github` 的 `feat/hls-ts-segments` 正由另一个会话运行整集。禁止修改、检出、提交主目录文件，禁止停止或向该任务发信号；不得写入其日志 `/tmp/codex-hls-huangguo-8.log`。本树已有的 v7 代码作为独立基线保存，保留已有修改。

Vast 整集任务结束且确认 GPU 空闲之前，禁止同步代码、禁止运行任何 GPU 作业、禁止修改远端 `/workspace/video-dewatermark`，禁止启动或重启 worker。本次任务全程不执行远端基准、不部署。下面部署说明保留作后续参考，不构成本次授权。

## 提速命令与效果对照

本地检查：`INTEGRATION= .venv/bin/python -m pytest worker/tests -q --tb=short`。没有虚拟环境时可只读调用主目录 Python，但设置 `PYTHONDONTWRITEBYTECODE=1`，避免向主目录写缓存。

基准入口：`.venv/bin/python -m scripts.hls_speedup_bench --help`；完整参数、独立目录的部署及 GPU 空闲后的操作见 `docs/speedup-v2.md`。本次不得在 Vast 执行。基准必须区分预期提速与实测结果，记录检测、分割、补画、合成、细节融合和编码的耗时；复用缓存必须校验输入和参数。

正式整集入口 `worker.hls.execute_hls` 默认使用 `speedup-v2` 原生路径。接口参数 `hls_native_path: "v7"` 或环境开关 `HLS_NATIVE_PATH=v7` 可一键回退修复与合成算法，同时保持1080p60交付；显式接口参数优先于环境默认。回退必须覆盖所有提速环境变量，不能混用两条路径的检查点。`hls_encoder` 支持 `libx264` 或 `h264_nvenc`，首次对照保留前者，回退固定使用前者。整集与基准共用 `worker.native_chunk.process_chunk`，不另写单片处理逻辑。

提速入口为流水线选项 `speedup_v2=True`，也可分别使用 `PROPAINTER_ROI=true`、`SAM_SPEEDUP=true`。原生脚本使用 `--backend torch-cuda --fused-detail --output-fps 60 --crf 10`；首次对照使用软件编码，显卡编码须先比较质量。基准必须在独立目录运行，使用固定模型清单，单块输入含上下文不超过30秒。标准测试依赖见 `worker/requirements-dev.txt`；完整张量对齐测试另需 `worker/requirements-speedup.txt`，本地使用纯处理器构建，不运行真实模型。

对照已接受 v7，保留原始残留候选及复核依据；残留为零、无修复闪现/闪烁是硬门槛。记录全部清晰度样本及低于 0.7 的样本；相对 v7 的清晰度、SSIM/PSNR 不得明显变差。没有 GPU 基准和目检结果不得宣称达到三倍提速或质量验收通过。输出 1920×1080、60 帧，音频只复制。

## Vast 连接与部署

主机 `175.121.93.64`，端口 `61586`，私钥路径 `/home/box/.ssh/vastai_ed25519`。

```sh
ssh -i /home/box/.ssh/vastai_ed25519 -p 61586 -o IdentitiesOnly=yes -o ConnectTimeout=10 root@175.121.93.64
```

本地仓库 `/workspace/video-dewatermark-github`；远端部署 `/workspace/video-dewatermark`。只同步代码与测试，不同步环境配置、密钥、视频、模型及本地虚拟环境。推荐先用 `tar` 打包所改的 `worker/`、`scripts/` 等代码文件，通过上述 SSH 的标准输入写入远端临时压缩包，校验成功后在部署目录解包。禁止使用带删除参数的全目录同步覆盖远端数据。

远端运行前加载配置（不要打印变量）：

```sh
set -a
. /opt/dewatermark/config/runtime.env
set +a
cd /workspace/video-dewatermark
export PYTHONPATH=.
```

主环境为 `.venv/bin/python`；YOLO、SAM、ProPainter 分别使用配置中的 `DETECT_PYTHON`、`TRACK_PYTHON`、`INPAINT_PYTHON`。用 `supervisorctl status dewatermark-worker` 检查，`supervisorctl start dewatermark-worker` 启动，`supervisorctl restart dewatermark-worker` 重启。先检查 GPU 占用和遗留核验进程，不能同时运行多份占满显存的补画任务。

## 素材、单块与整集

整集通过网页操作台或 API 提交对象键 `kelin/hls-test/huangguo-sangni1-ep1/index.m3u8`。API 在远端 `127.0.0.1:18080`，网页操作台在 `https://175.121.93.64:61591/`。认证信息只在内存读取，不输出。明文源冻结在 `/workspace/dewatermark-data/hls-huangguo/source.mp4`。

单块通用处理入口：

```sh
.venv/bin/python -m scripts.hls_chunk_check --source /workspace/dewatermark-data/hls-huangguo/source.mp4 --prior /工作目录/multitrack-prior.npz --output /验收目录 --duration 25 --context 2
```

原生合成、细节重合成、编码后核验入口为 `scripts.hls_native_chunk`、`scripts.hls_recompose_detail`、`scripts.hls_native_verify`、`scripts.hls_brand_verify`、`scripts.hls_detail_verify`，各模块 `--help` 列出所需中间文件。整集主入口为 API 任务调用 `worker.hls.execute_hls`；导出已成功任务使用：

```sh
.venv/bin/python -m scripts.hls_export --job 任务编号 --output /验收目录 --detect
```

kelin 已接受远端 `/workspace/dewatermark-data/hls-huangguo/chunk1-v7/chunk1.mp4`，2026年11月11日07:18越南时间生成，1920×1080、30帧。整集沿用 v7：多轨迹与按镜头/生命周期初始化 SAM、稀疏真实笔画掩膜、完整品牌图标覆盖、36帧窗口/8帧重叠的补画、原生干净参考纹理、标准差3的多频带联合纹理投影，最终细节合成 CRF 10。停止第1块清晰度调参，不采用之后未接受的 v8、标准差6、透明反解等试验。

整集输出必须为1920×1080、60帧；模型处理中间帧率可以为30帧。音频只复制，不重编码。按原分片边界规划主体块，前后带上下文，单块总输入不得超过30秒。单卡 GPU 推理安排有界队列；可并行准备、合成及核验，不能以降质换速度。失败只重跑失败块，保留已完成检查点。

本地单块产物在 `/workspace/user-videos/hls-huangguo/chunk1/`；整集产物在 `/workspace/user-videos/hls-huangguo/full/`。远端中间文件在 `/workspace/dewatermark-data/hls-huangguo/`。每块完成立即向本地整集目录 `progress.log` 追加块号、耗时、残留及清晰度最低值。原 R2 对象不改，结果写不同前缀。

## 验收报告

`report.json` 必须记录各块及总耗时、所有清晰度样本、最低值、低于0.7的样本清单、原始残留候选及复核依据、接缝指标、输出规格、时长、音频压缩数据一致性、播放检查与目检结果，以及安全的 R2 对象前缀。

最新门槛自2026年11月11日09:56越南时间生效：**清晰度仅记录，不是硬门槛**，不得因低于0.8判失败或返工；低于0.7必须单独列出。**水印残留必须为0、无修复造成的闪现/闪烁仍是硬门槛**。原片镜头切换应结合源片作比较。`通过` 与 `目检` 均通过才能宣称验收通过；不能篡改原始候选来制造零残留。字幕不能当水印。

必须生成整集成品 MP4（新 HLS 无损合并）、左右并排对比视频，以及5、60、120、200秒附近左上和右下对比图。验证新 HLS 可播放、分片时间轴与接缝、原片时长、音画同步、音频复制。未完成检查要如实记录，不能写成已通过。

## 测试

本地常用命令：

```sh
INTEGRATION= .venv/bin/python -m pytest worker/tests -q --tb=short
GOFLAGS= go test ./...
```

或运行 `make test`。需要连接实际数据库与存储的集成测试在配置齐全且隔离的数据环境运行 `INTEGRATION=1 .venv/bin/python -m pytest worker/tests -q`。每项通用修复补充针对行为的单元测试。最终记录实际通过、跳过及失败数，检查完成后本地提交，绝不推送。
