# 模型 provenance 与生产阻断

2026-11-09 仅访问官方 JSON metadata 和源码文本；未下载 weights 或数据集。

- [SAM 源码 commit](https://github.com/facebookresearch/sam2/commit/2b90b9f5ceec907a1c18123530e92e794ad901a4)。
- [Small 官方 HF metadata](https://huggingface.co/api/models/facebook/sam2.1-hiera-small?blobs=true)，revision `ee5bba1d82bb8749febdf90f45e84b687142ba03`，`sam2.1_hiera_small.pt` SHA-256 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`。
- [Tiny 官方 HF metadata](https://huggingface.co/api/models/facebook/sam2.1-hiera-tiny?blobs=true)，revision `de431c4043854a71d8101e17995dfe596bf101a5`，`sam2.1_hiera_tiny.pt` SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。
- [ProPainter source commit](https://github.com/sczhou/ProPainter/commit/326782529a63d997cab80ab21439b92ed422b3d3)，[官方 release](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0)，[release metadata](https://api.github.com/repos/sczhou/ProPainter/releases/tags/v0.1.0)。三项必需制品 `ProPainter.pth`、`raft-things.pth`、`recurrent_flow_completion.pth` 的 `digest` 均为 null。manifest 明确 UNRESOLVED，禁止生产启动。
- [DiffuEraser 源码](https://github.com/lixiaowen-xw/DiffuEraser) 与 [HF](https://huggingface.co/lixiaowen/diffuEraser/tree/main)：phase2，revision/checksum 未冻结，不在本版执行。

生产校验完整遍历必需资产（包括 Tiny fallback），校验本地 SHA-256 与实际源码 revision/clean 状态；ProPainter 真正使用的 source/weights 路径另行校验。`worker/propainter_offline.py` 替换上游自动下载函数，只允许读取已存在的本地资产。

合并进本仓库后，`worker.main` 在 `INPAINT_ENGINE=propainter` 时先调用 `validate_provenance(deploy/models.json)`，digest 未公布就退出；通过之后才核对操作员 lock，并要求 lock 覆盖清单中的每个 digest。`telea-dev` / `mvp-1-dev` 不走这套校验，只用于工程冒烟，不声称运行了 SAM/ProPainter。

容器基础镜像/服务使用版本 tag，尚未解析平台专属 digest；上线前需填入审查过的 immutable image digest 并保存 GPU 依赖 lock。当前候选 GPU 依赖不是兼容性验收结果。
