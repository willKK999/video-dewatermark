"""统一将数值计算产物转为JSON原生值；保持数组维度、数值和候选内容。"""
from collections.abc import Mapping
import numbers


def json_native(value):
    value_type = type(value)
    if value_type in (str, int, float, bool, type(None)):
        return value
    if value_type is dict or isinstance(value,Mapping):
        return {json_native(k):json_native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_native(v) for v in value]
    if value_type.__module__.partition('.')[0]=='numpy':
        if hasattr(value,'tolist') and getattr(value,'ndim',0)>0:
            return json_native(value.tolist())
        if isinstance(value,numbers.Integral):return int(value)
        if isinstance(value,numbers.Real):return float(value)
        if hasattr(value,'item'):
            item=value.item()
            if type(item) is type(value):raise TypeError('无法转换的NumPy标量类型：'+type(value).__name__)
            return json_native(item)
    return value
