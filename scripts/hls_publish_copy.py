"""视频去水印系统：把成功产物复制到独立安全前缀，不输出内部租约标识。"""

import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from worker import store
from worker.main import object_store


def copy_result(job_id, prefix, output):
    if not prefix.endswith("/"):
        raise ValueError("交付前缀必须以斜线结尾")
    with store.connect(os.environ["DATABASE_URL"]) as db:
        row = db.execute(
            "SELECT tenant_id,input_key,status,result_key,result_meta FROM jobs WHERE id=%s",
            (job_id,),
        ).fetchone()
    if not row or row["status"] != "SUCCEEDED":
        raise ValueError("只能复制已成功任务的交付文件")
    input_prefix = row["input_key"].rsplit("/", 1)[0] + "/"
    if not prefix.startswith(row["tenant_id"] + "/") or prefix == input_prefix:
        raise ValueError("交付前缀必须属于本租户，且与原输入前缀不同")
    s3 = object_store()
    bucket = os.environ["S3_BUCKET"]
    source = row["result_key"].rsplit("/", 1)[0] + "/"
    if prefix == source or prefix.startswith(source):
        raise ValueError("交付前缀必须与内部产物分开")
    # 检查点与内部租约前缀仅在内存使用，不写入对外报告。
    objects = [
        obj
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=source
        )
        for obj in page.get("Contents", [])
    ]
    objects = [obj for obj in objects if obj["Key"][len(source) :] != "source.mp4"]

    def copy(obj):
        relative = obj["Key"][len(source) :]
        key = prefix + relative
        s3.copy_object(
            Bucket=bucket, Key=key, CopySource={"Bucket": bucket, "Key": obj["Key"]}
        )
        if s3.head_object(Bucket=bucket, Key=key)["ContentLength"] != obj["Size"]:
            raise ValueError("复制后对象大小不一致")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy, objects))
    output.mkdir(parents=True, exist_ok=True)
    safe = {
        "前缀": prefix,
        "播放列表对象键": prefix + "hls/index.m3u8",
        "成品对象键": prefix + "result.mp4",
        "打包对象键": prefix + "hls.zip",
        "对象数": len(objects),
    }
    (output / "结果前缀.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2)
    )
    print("已复制到独立结果前缀：" + prefix + "，对象数：" + str(len(objects)))
    return safe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    copy_result(a.job, a.prefix, a.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("独立前缀复制失败：" + type(exc).__name__)
        raise SystemExit(1)
