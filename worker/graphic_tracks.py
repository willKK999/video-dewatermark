"""独立图形的逐镜头关联；新对象或明显形态变化不能沿用旧跟踪状态。"""

from .short_brand import fraction_inside
from .yolo_stage import compatible_scale


def compatible(a, b):
    return (
        a["class_id"] == b["class_id"]
        and compatible_scale(a["box"], b["box"])
        and fraction_inside(a["box"], b["box"]) > 0.5
        and fraction_inside(b["box"], a["box"]) > 0.5
    )


def associate(rows, scenes):
    result = []
    for start, end in scenes:
        tracks = []
        for row in rows:
            frame = row["frame"]
            if not start <= frame < end:
                continue
            old_count = len(tracks)
            used, created = set(), []
            for point in row["boxes"]:
                options = [
                    k
                    for k, track in enumerate(tracks)
                    if k not in used
                    and track.get("closed") is None
                    and frame - track["points"][-1]["frame"] <= 30
                    and compatible(point, track["points"][-1])
                ]
                if options:
                    k = max(
                        options,
                        key=lambda k: fraction_inside(
                            point["box"], tracks[k]["points"][-1]["box"]
                        ),
                    )
                    tracks[k]["points"].append(dict(point, frame=frame))
                else:
                    k = len(tracks)
                    tracks.append(dict(points=[dict(point, frame=frame)]))
                    created.append(k)
                used.add(k)
            # 同位置显著改变形态且旧对象不再被观测时，终止旧状态的尾部上下文。
            # 同帧仍有旧对象的独立观测时，两条轨迹都保留。
            for k in range(old_count):
                track = tracks[k]
                if k in used or track.get("closed") is not None:
                    continue
                previous = track["points"][-1]
                if frame - previous["frame"] > 30:
                    continue
                for new in created:
                    point = tracks[new]["points"][0]
                    if (
                        not compatible(previous, point)
                        and max(
                            fraction_inside(previous["box"], point["box"]),
                            fraction_inside(point["box"], previous["box"]),
                        )
                        > 0.5
                    ):
                        track["closed"] = frame
                        break
        for track in tracks:
            points = track["points"]
            finish = min(end, points[-1]["frame"] + 31, track.get("closed", end))
            result.append(dict(scenes=[[points[0]["frame"], finish]], hints=points))
    return result
