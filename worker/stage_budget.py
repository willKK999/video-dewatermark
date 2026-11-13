"""按实际SAM调度工作量估算执行预算；不改变旧调用的900秒默认值。"""
from dataclasses import asdict, dataclass
import math
from .sam_schedule import groups_for_scene, sampled_hints


@dataclass(frozen=True)
class BudgetSettings:
    minimum_seconds: int = 900
    maximum_seconds: int = 7200
    model_load_seconds: float = 90
    initialization_seconds: float = 8
    decoded_frame_seconds: float = .05
    object_frame_seconds: float = .09
    prompt_seconds: float = .25
    reference_pixels: int = 1280 * 720

    def __post_init__(self):
        if not 0 < self.minimum_seconds <= self.maximum_seconds <= 86400:
            raise ValueError('执行预算上下界无效')


def sam_workload(detections, frame_count, pixels):
    groups = [g for a, b in detections['scenes']
              for g in groups_for_scene(detections, a, b)]
    return dict(
        frames=frame_count, pixels=pixels,
        tracks=len({p.get('track_id', 1) for p in detections['hints']}),
        boxes=len(detections['hints']), initializations=len(groups),
        decoded_frames=sum(b-a for a,b,h in groups),
        object_frames=sum((b-a+1)*len({p.get('track_id',1) for p in h})
                          for a,b,h in groups),
        sampled_prompts=sum(len(sampled_hints(h,a,b)) if detections.get('multitrack') else len(h)
                            for a,b,h in groups),
    )


def sam_budget(detections, frame_count, pixels, settings=None):
    cfg = settings or BudgetSettings()
    work = sam_workload(detections, frame_count, pixels)
    scale = max(1., pixels/cfg.reference_pixels)
    estimate = (cfg.model_load_seconds
                + work['initializations']*cfg.initialization_seconds
                + scale*(work['decoded_frames']*cfg.decoded_frame_seconds
                         + work['object_frames']*cfg.object_frame_seconds
                         + work['sampled_prompts']*cfg.prompt_seconds))
    seconds = min(cfg.maximum_seconds,max(cfg.minimum_seconds,math.ceil(estimate)))
    return dict(stage='mask',timeout_seconds=seconds,estimated_seconds=estimate,
                upper_bound_reached=estimate>cfg.maximum_seconds,
                workload=work,settings=asdict(cfg),method='sam-workload-1')
