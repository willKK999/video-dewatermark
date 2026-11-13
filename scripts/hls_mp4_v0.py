"""本地明文 MP4 的通用单轮整集调度；复用稳定模型流程，失败继续且不宣称验收通过。"""

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from worker import media, vision
from worker.hls import chunks, seam_metrics
from worker.hls_acceptance import clarity_summary
from worker.hls_settings import HLSSettings
from worker.pipeline import execute
from worker.process import run
from scripts.hls_chunk_check import audio_hash, cut_references


@dataclass(frozen=True)
class AuditSettings:
    # 与稳定整集最终核验保持相同的时域候选规则。
    extra_mae: float = 6
    change_ratio: float = 1.8
    repair_mae: float = 12


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def log(root, message):
    line = time.strftime("%Y-%m-%dT%H:%M:%S%z") + " " + message
    with (root / "out/progress.log").open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def initialize(root, expected):
    source = root / "source.mp4"
    digest = vision.sha256(source)
    if digest != expected:
        raise ValueError("源文件摘要不匹配，禁止继续")
    v = next(s for s in media.probe(source)["streams"] if s["codec_type"] == "video")
    source_fps = float(Fraction(v["avg_frame_rate"]))
    fps = min(source_fps, HLSSettings.from_env().model_fps)
    total = int(v.get("nb_frames") or round(float(v["duration"]) * source_fps))
    data = dict(源摘要=digest, 源帧率=source_fps, 处理帧率=fps,
                源尺寸=[v["width"], v["height"]], 源帧数=total,
                视频时长秒=float(v["duration"]),
                计划=chunks([total / source_fps], fps),
                配置=asdict(HLSSettings.from_env()), 时域规则=asdict(AuditSettings()))
    if (root / "run.json").exists():
        if read(root / "run.json") != data:
            raise ValueError("已有整集检查点与本次输入或配置不一致")
    else:
        write(root / "run.json", data)
    return data


def select_frames(work, count, destination):
    """缺少模型帧时明确失败，禁止把原帧静默混入成片。"""
    missing = [i for i in range(count) if not (work / "composite" / f"{i:06d}.png").is_file()]
    if missing:
        from worker.process import PipelineError
        raise PipelineError("OUTPUT_INCOMPLETE", "模型输出帧不完整，禁止回退原帧",
                            details=dict(missing_count=len(missing), missing_frames=missing))
    destination.mkdir()
    missing = []
    for i in range(count):
        name = f"{i:06d}.png"
        path = work / "composite" / name
        if not path.exists():
            raise ValueError("模型帧已丢失，禁止回退原帧")
        os.link(path, destination / name)
    return missing


def validated_prior(root, source_digest):
    """内容缓存必须同时绑定源片及先验摘要，不能借用其他输入的先验。"""
    report = read(root / "prior-report.json")
    if report.get("通过") is not True:
        return None
    path = root / "multitrack-prior.npz"
    if (report.get("源摘要") != source_digest
            or not path.is_file()
            or vision.sha256(path) != report.get("先验摘要")):
        raise ValueError("自动先验缓存与本片内容摘要不一致")
    return path


def cropped_references(detected, first, last):
    if detected.get("tracks") is not None:
        return cut_references(detected, first, last)
    return dict(detected,
                scenes=[[max(a, first)-first, min(b,last)-first]
                        for a,b in detected["scenes"] if a < last and b > first],
                hints=[dict(p, frame=p["frame"]-first)
                       for p in detected["hints"] if first <= p["frame"] < last])


def audit_block(root, work, video, plan, fps, detected):
    """编码后复检保留全部原始证据；逐帧时域比较源片，清晰度只记录。"""
    from worker.multitrack import load_prior, residual_quality, temporal_consensus
    from worker.low_frequency import band, fixed_scores
    from worker.matte import temporal_detail

    started = time.monotonic()
    first, count = plan["start"]-plan["left"], plan["end"]-plan["start"]
    refs = cropped_references(detected, first, first+count)
    audit = work / "encoded-audit"
    audit.mkdir()
    write(audit / "detections.json", refs)
    source = cv2.VideoCapture(str(root / "source.mp4"))
    result = cv2.VideoCapture(str(video))
    source.set(cv2.CAP_PROP_POS_MSEC, plan["start"] * 1000 / fps)
    settings = AuditSettings()
    hints = {}
    for p in refs["hints"]:
        hints.setdefault(p["frame"], []).append(p)
    cuts = {s for s,e in refs["scenes"]}
    all_time, flashes, clarity, lows = [], [], [], []
    prior = load_prior(Path(refs["prior"])) if refs.get("multitrack") else None
    previous = None

    def reader(i):
        nonlocal previous
        good, x = source.read()
        ok, y = result.read()
        if not good or not ok or x.shape != y.shape:
            raise ValueError("编码后逐帧核验帧数或尺寸不一致")
        mask_path = work / "masks" / f"{i+first:06d}.png"
        mask = cv2.imread(str(mask_path), 0) if mask_path.exists() else None
        if prior and i % max(1, round(fps)) == 0:
            lows.append(band(cv2.resize(y, (prior[0].shape[1], prior[0].shape[0]))))
        if previous is not None:
            old_x, old_y, old_mask = previous
            ds = np.abs(x.astype(np.float32)-old_x)
            dy = np.abs(y.astype(np.float32)-old_y)
            dr = np.abs((y.astype(np.float32)-x)-(old_y.astype(np.float32)-old_x))
            for point in hints.get(i, []):
                a,b,c,d = map(int, point["box"])
                a,b,c,d=max(0,a),max(0,b),min(x.shape[1],c),min(x.shape[0],d)
                if a >= c or b >= d:
                    continue
                item = dict(frame=i, time=(plan["start"]+i)/fps,
                            track_id=point.get("track_id"), box=point["box"],
                            source_mae=float(ds[b:d,a:c].mean()),
                            result_mae=float(dy[b:d,a:c].mean()),
                            repair_delta_mae=float(dr[b:d,a:c].mean()),
                            source_scene_cut=i in cuts)
                all_time.append(item)
                if (i not in cuts and item["result_mae"] > max(
                    item["source_mae"]+settings.extra_mae,
                    item["source_mae"]*settings.change_ratio)
                    and item["repair_delta_mae"] > settings.repair_mae):
                    flashes.append(item)
            if (i % max(1, round(fps)) in (0,1) and mask is not None
                    and old_mask is not None):
                metric = temporal_detail(old_x,x,old_y,y,mask & old_mask,mask | old_mask)
                metric.pop("passed", None)
                metric.pop("threshold", None)
                clarity.append(dict(frame=i, time=(plan["start"]+i)/fps, **metric))
        previous = x, y, mask
        return y

    try:
        if prior:
            try:
                residual_quality(audit, [Path(f"{i:06d}.png") for i in range(count)],
                                 refs, frame_reader=reader)
            except Exception as exc:
                if getattr(exc,"code",None) != "RESIDUAL_UNCERTAIN":
                    raise
        else:
            # 单轨迹分支也执行完整编码后 YOLO 复检，候选不自动消除。
            encoded = audit / "composite"
            encoded.mkdir()
            for i in range(count):
                cv2.imwrite(str(encoded / f"{i:06d}.png"), reader(i))
            try:
                vision.residual_quality(audit, sorted(encoded.glob("*.png")), refs)
            except Exception as exc:
                if getattr(exc,"code",None) != "RESIDUAL_UNCERTAIN":
                    raise
    finally:
        source.release()
        result.release()
    low = None
    if prior and lows:
        samples = sorted((root / "consensus-samples").glob("*.jpg"))
        # 使用同一内容自动派生的全片低频共识，绝不复用旧片模板。
        source_bands=[band(cv2.imread(str(p))) for p in samples]
        template=np.median(np.stack(source_bands),axis=0)
        low=fixed_scores(template,np.median(np.stack(lows),axis=0),prior[2])
    quality = read(audit / "quality.json")
    data = dict(视频摘要=vision.sha256(video), 编码后残留=quality,
                低频复检=low, 低频残留候选数=(sum(p["residual"] for p in low)
                                           if low is not None else None),
                闪烁候选=flashes, 时域全部样本=all_time,
                清晰度=clarity_summary(clarity), 耗时秒=time.monotonic()-started,
                目检=dict(通过=False,状态="尚未完成完整视频目检"), 通过=False,
                说明="候选保留原值；自动零候选不能替代完整图形及整集目检。清晰度为掩膜内时域细节指标，仅记录。")
    write(audit / "report.json", data)
    return data


def process_block(root, scratch, index, data):
    started = time.monotonic()
    plan = data["计划"][index-1]
    work = scratch / f"chunk-{index:02d}"
    output = root / "out" / f"chunk-{index:02d}"
    output.mkdir()
    work.mkdir()
    fps = data["处理帧率"]
    count = plan["right"]-plan["left"]
    phases, failures = {}, []
    current_stage = "切块"
    def stage(name):
        nonlocal current_stage
        current_stage = name
        log(root, f"块号={index} 阶段={name}")
    clip = work / "input.mp4"
    log(root, f"块号={index}/{len(data['计划'])} 开始")
    cut_start=time.monotonic()
    run(["ffmpeg","-v","error","-y","-threads","2","-ss",str(plan["left"]/fps),
         "-i",root/"source.mp4","-an","-vf",f"fps={fps}","-frames:v",str(count),
         "-c:v","libx264","-preset","ultrafast","-crf","0","-threads","2",clip])
    phases["切块"] = time.monotonic()-cut_start
    options = dict(hls=True, adaptive_timeouts=True)
    prior = validated_prior(root, data["源摘要"])
    if prior:
        options["multitrack_prior"] = str(prior)
    try:
        execute(clip,work,options,"propainter",True,stage)
    except Exception as exc:
        code = getattr(exc,"code",type(exc).__name__)
        failure = dict(阶段=current_stage,错误码=code,详情=getattr(exc,"details",{}))
        row = dict(块号=index,计划=plan,通过=False,目检通过=False,阶段失败=[failure],
                   阶段耗时秒={k:v/1000 for k,v in read(work/"phase_ms.json").items()},
                   原帧回退数=0,缺失模型输出=True,完整产物=None,
                   本轮墙时秒=time.monotonic()-started)
        write(output/"report.json",row)
        log(root, f"失败块={index} 失败阶段={current_stage} 错误码={code} 原帧回退=禁止 通过=False")
        raise
    phases.update({k:v/1000 for k,v in read(work/"phase_ms.json").items()})
    selected = work/"selected"
    if (work/"frames").exists() and len(list((work/"frames").glob("*.png"))) == count:
        missing = select_frames(work,count,selected)
        frame_input=["-framerate",str(fps),"-start_number",str(plan["start"]-plan["left"]),
                     "-i",selected/"%06d.png"]
    else:
        from worker.process import PipelineError
        log(root,f"失败块={index} 失败阶段=主体编码 错误码=OUTPUT_INCOMPLETE 原帧回退=禁止")
        raise PipelineError("OUTPUT_INCOMPLETE","缺少完整模型帧，禁止生成回退成片")
    video=output/"video.mp4"
    t=time.monotonic()
    run(["ffmpeg","-v","error","-y",*frame_input,"-an","-frames:v",
         str(plan["end"]-plan["start"]),"-c:v","libx264","-preset","fast",
         "-crf",str(HLSSettings.from_env().video_crf),"-threads","2","-pix_fmt","yuv420p",video])
    phases["主体编码"] = time.monotonic()-t
    audit={}
    if (work/"detections.json").exists():
        try:
            audit=audit_block(root,work,video,plan,fps,read(work/"detections.json"))
            phases["编码后核验"]=audit["耗时秒"]
        except Exception as exc:
            failures.append(dict(阶段="编码后核验",错误码=getattr(exc,"code",type(exc).__name__)))
    row=dict(块号=index,计划=plan,源摘要=data["源摘要"],版本="0e4dc7b通用流程单轮",
             视频摘要=vision.sha256(video),阶段耗时秒=phases,本轮墙时秒=time.monotonic()-started,
             阶段失败=failures,原帧回退帧号=missing,原帧回退数=len(missing),核验=audit,
             基础核验=read(work/"quality.json"),模型窗口=read(work/"performance.json"),
             通过=False,目检通过=False,
             说明="稳定基础流程仅运行一轮；不套用旧片专属资产，不自动进入逐块多轮补修。缺失模型输出明确回退原帧。")
    write(output/"report.json",row)
    log(root,f"块号={index}/{len(data['计划'])} 耗时秒={row['本轮墙时秒']:.3f} "
        f"残留={audit.get('编码后残留',{}).get('residual_count')} "
        f"清晰度最低值={audit.get('清晰度',{}).get('最小保留比')} 原帧回退数={len(missing)} 通过=False")


def deliver(root,data):
    started=time.monotonic()
    output=root/"out"
    videos=[output/f"chunk-{i+1:02d}/video.mp4" for i in range(len(data["计划"]))]
    listing=output/"concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in videos))
    silent=output/"joined-video.mp4"
    run(["ffmpeg","-v","error","-y","-f","concat","-safe","0","-i",listing,
         "-c","copy",silent])
    result=output/"整集-v0.mp4"
    original=media.probe(root/"source.mp4")
    original_video=next(s for s in original["streams"] if s["codec_type"]=="video")
    video_offset=(float(original_video.get("start_time",0))
                  -float(original["format"].get("start_time",0)))
    # 不限制音轨时长、不使用 shortest，完整复制原片全部 AAC 包。
    run(["ffmpeg","-v","error","-y","-itsoffset",str(video_offset),
         "-i",silent,"-i",root/"source.mp4",
         "-map","0:v:0","-map","1:a:0?","-c","copy","-movflags","+faststart",result])
    comparison=output/"对比-原片vs整集-v0.mp4"
    run(["ffmpeg","-v","error","-y","-threads","2","-i",root/"source.mp4","-i",result,
         "-filter_complex_threads","2","-filter_complex","[0:v][1:v]hstack=inputs=2[v]",
         "-map","[v]","-map","0:a:0?","-c:v","libx264","-preset","fast","-crf","18",
         "-threads","2","-c:a","copy","-movflags","+faststart",comparison])
    meta=media.probe(result,count=True)
    v=next(s for s in meta["streams"] if s["codec_type"]=="video")
    equality=audio_hash(root/"source.mp4")==audio_hash(result)==audio_hash(comparison)
    frames=int(v["nb_read_frames"])
    technical=([v["width"],v["height"]]==data["源尺寸"] and
               float(Fraction(v["avg_frame_rate"]))==data["源帧率"] and
               frames==data["源帧数"] and equality)
    for path in [result,comparison]:
        run(["ffmpeg","-v","error","-xerror","-threads","2","-i",path,"-f","null","-"])
    records=[read(output/f"chunk-{i+1:02d}/report.json") for i in range(len(videos))]
    seams=seam_metrics(root/"source.mp4",output,data["计划"],data["处理帧率"],
                       output,lambda:None,result_path=result)
    all_clarity=[p for r in records for p in r.get("核验",{}).get("清晰度",{}).get("全部样本",[])]
    report=dict(系统="视频去水印系统",版本="整集v0",稳定基线="0e4dc7b",
                源=data,流程完成=True,通过=False,目检=dict(通过=False,状态="待完整目检"),
                各块=records,未通过块=[r["块号"] for r in records],
                清晰度=clarity_summary(all_clarity),接缝=seams,
                输出规格=dict(尺寸=[v["width"],v["height"]],帧率=float(Fraction(v["avg_frame_rate"])),
                              帧数=frames,视频时长秒=float(v["duration"]),封装时长秒=float(meta["format"]["duration"])),
                音频压缩数据一致=equality,技术检查通过=technical,完整解码通过=True,
                拼接与交付耗时秒=time.monotonic()-started,
                源摘要=data["源摘要"],成品摘要=vision.sha256(result),
                对比摘要=vision.sha256(comparison),R2对象前缀=None,
                播放检查=dict(状态="待浏览器播放检查",通过=False),
                说明="v0流程交付不等于验收通过。未处理原帧、残留、低频及闪烁候选原样保留；字幕不自动作为水印。未上传对象存储。")
    write(output/"report.json",report)
    log(root,"整集 v0 拼接与完整解码完成；验收尚未通过")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--scratch",type=Path)
    p.add_argument("--sha256",required=True)
    p.add_argument("--block",type=int)
    p.add_argument("--deliver",action="store_true")
    a=p.parse_args()
    cv2.setNumThreads(2)
    if os.environ.get("DETECTOR") != "yolo":
        raise ValueError("本入口只接受自动 YOLO 检测配置")
    data=initialize(a.root,a.sha256)
    if a.block:
        if not a.scratch or not 1 <= a.block <= len(data["计划"]):
            p.error("需要合法块号及临时目录")
        process_block(a.root,a.scratch,a.block,data)
    elif a.deliver:
        deliver(a.root,data)
    else:
        print(json.dumps(data,ensure_ascii=False))


if __name__ == "__main__":
    main()
