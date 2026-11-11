"""原生块的可移植检查点；先上传完整归档，最后提交内容散列标记。"""

import json
import re
import zipfile
from pathlib import Path
from botocore.exceptions import ClientError
from .process import PipelineError
from .roi_inpaint import digest


class BundleStore:
    def __init__(self, s3, bucket, prefix, check=lambda: None):
        self.s3, self.bucket, self.prefix, self.check = (
            s3,
            bucket,
            prefix.rstrip("/"),
            check,
        )

    def save(self, name, root, paths, identity):
        self.check()
        archive = root / (name + ".checkpoint.zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as out:
            entries = {}
            for path in paths:
                path = Path(path)
                for file in sorted(path.rglob("*")) if path.is_dir() else [path]:
                    self.check()
                    if file.is_file() and not file.is_symlink():
                        relative = file.relative_to(root).as_posix()
                        if not allowed(relative):
                            raise PipelineError(
                                "OUTPUT_INVALID", "检查点出现非生成文件"
                            )
                        entries[relative] = digest(file)
                        out.write(file, relative)
            out.writestr(
                "bundle.json", json.dumps(dict(identity=identity, files=entries))
            )
        checksum = digest(archive)
        key = self.prefix + "/" + name + "-" + checksum + ".zip"
        self.s3.upload_file(
            str(archive), self.bucket, key, Callback=lambda _: self.check()
        )
        self.check()
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.prefix + "/" + name + ".json",
            Body=json.dumps(dict(identity=identity, sha256=checksum)).encode(),
            ContentType="application/json",
        )
        archive.unlink()

    def restore(self, name, root, identity):
        self.check()
        try:
            stream = self.s3.get_object(
                Bucket=self.bucket, Key=self.prefix + "/" + name + ".json"
            )["Body"]
            try:
                marker = json.loads(stream.read(65536))
            finally:
                stream.close()
            if marker.get("identity") != identity or not re.fullmatch(
                r"[0-9a-f]{64}", marker.get("sha256", "")
            ):
                return False
            archive = root / (name + ".restore.zip")
            self.s3.download_file(
                self.bucket,
                self.prefix + "/" + name + "-" + marker["sha256"] + ".zip",
                str(archive),
                Callback=lambda _: self.check(),
            )
        except ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        try:
            if digest(archive) != marker["sha256"]:
                return False
            with zipfile.ZipFile(archive) as bundle:
                members = bundle.infolist()
                if (
                    sum(m.file_size for m in members) > 16 * 1024**3
                    or len(members) > 20000
                ):
                    raise PipelineError("OUTPUT_INVALID", "检查点归档超出有界上限")
                meta = json.loads(bundle.read("bundle.json"))
                names = [m.filename for m in members if m.filename != "bundle.json"]
                if (
                    meta.get("identity") != identity
                    or not names
                    or set(names) != set(meta["files"])
                    or len(names) != len(set(names))
                    or any(not allowed(n) for n in names)
                ):
                    raise PipelineError("OUTPUT_INVALID", "检查点文件列表或路径无效")
                # 先验证全部文件，再写入目标，不能混合新旧不完整归档。
                import hashlib

                for name in names:
                    self.check()
                    with bundle.open(name) as stream:
                        checksum = (
                            hashlib.file_digest(stream, "sha256").hexdigest()
                            if hasattr(hashlib, "file_digest")
                            else hash_stream(stream)
                        )
                    if checksum != meta["files"][name]:
                        raise PipelineError("OUTPUT_INVALID", "检查点成员摘要不一致")
                for name in names:
                    self.check()
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.is_symlink() or any(
                        p.is_symlink() for p in path.parents if p != root.parent
                    ):
                        raise PipelineError("OUTPUT_INVALID", "检查点目标含符号链接")
                    with bundle.open(name) as source, path.open("wb") as target:
                        import shutil

                        shutil.copyfileobj(source, target, 1024 * 1024)
            return True
        except (zipfile.BadZipFile, ValueError, KeyError):
            raise PipelineError("OUTPUT_INVALID", "检查点归档损坏") from None
        finally:
            archive.unlink(missing_ok=True)


def hash_stream(stream):
    import hashlib

    checksum = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        checksum.update(block)
    return checksum.hexdigest()


def allowed(name):
    path = Path(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.name != "key.bin"
        and path.suffix in {".json", ".npz", ".npy", ".png", ".mkv"}
        and "frames" not in path.parts
        and "native-reference" not in path.parts
        and "stages" not in path.parts
    )
