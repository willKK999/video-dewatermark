"""视频去水印系统 HLS 浏览器验收；凭证从文件读取，不记录网络地址。"""

import argparse
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--url", default="https://175.121.93.64:61591/")
p.add_argument("--input", default="kelin/hls-test/20261110/input.m3u8")
p.add_argument("--output", default="/workspace/user-videos/hls")
p.add_argument("--token-file", default="/home/box/.config/dewatermark/console-token")
p.add_argument("--existing", action="store_true")
p.add_argument("--retry", action="store_true")
p.add_argument(
    "--submit-only", action="store_true", help="只提交并保存任务编号，不等待整集结束"
)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as pw:
    browser = pw.chromium.launch(
        executable_path="/opt/google/chrome/chrome",
        headless=True,
        args=["--no-sandbox"],
    )
    context = browser.new_context(
        ignore_https_errors=True,
        viewport={"width": 1440, "height": 1100},
        accept_downloads=True,
        locale="zh-CN",
    )
    page = context.new_page()
    page.goto(a.url, wait_until="domcontentloaded")
    page.locator("#token").fill(Path(a.token_file).read_text().strip())
    page.locator("#loginForm button").click()
    page.locator("#workspace").wait_for(state="visible")
    if not a.existing:
        page.locator("#hlsInput").fill(a.input)
        page.locator("#hlsSubmit").click()
        page.wait_for_function(
            "() => document.getElementById('uploadStatus').textContent.includes('HLS 任务已提交')"
        )
        job = page.evaluate("selectedJob")
        (out / "job.json").write_text(json.dumps({"id": job}))
        print("浏览器已提交 HLS 任务：" + job, flush=True)
    else:
        job = json.loads((out / "job.json").read_text())["id"]
        page.evaluate("id=>selectJob(id)", job)
        if a.retry:
            page.locator("#" + job).get_by_role("button", name="重试未完成块").click()
            page.wait_for_function(
                "id=>['QUEUED','RUNNING'].includes(document.querySelector('#'+id+' .badge')?.dataset.state)",
                arg=job,
            )
            print("已通过操作台重试未完成块", flush=True)
    if a.submit_only:
        page.screenshot(path=str(out / "console-submitted.png"), full_page=True)
        browser.close()
        raise SystemExit(0)
    previous = None
    for _ in range(8640):
        try:
            data = page.evaluate(
                "async id=>await Promise.race([api('/v1/jobs/'+id).then(r=>r.json()),new Promise((_,reject)=>setTimeout(()=>reject(Error('状态请求超时')),25000))])",
                job,
            )
        except Exception:
            print("进度读取暂时失败，重新连接操作台", flush=True)
            page.reload(wait_until="domcontentloaded", timeout=30000)
            page.locator("#workspace").wait_for(state="visible")
            page.evaluate("id=>selectJob(id)", job)
            continue
        state = (data["status"], data.get("stage"), json.dumps(data.get("progress")))
        if state != previous:
            print("任务进度：" + json.dumps(state, ensure_ascii=False), flush=True)
            previous = state
            page.screenshot(path=str(out / "console-progress.png"), full_page=True)
        if data["status"] == "SUCCEEDED":
            break
        if data["status"] in ("FAILED", "NEEDS_REVIEW", "CANCELED"):
            raise RuntimeError("HLS 任务未通过：" + str(data.get("error", {})))
        page.wait_for_timeout(5000)
    else:
        raise RuntimeError("HLS 验收等待超时")
    page.evaluate("refresh()")
    page.wait_for_timeout(1000)
    page.evaluate("id=>selectJob(id)", job)
    page.locator("#" + job).get_by_role("button", name="播放对比与下载").click()
    page.wait_for_function(
        "() => document.querySelectorAll('video').length===2 && [...document.querySelectorAll('video')].every(v=>v.readyState>=2)",
        timeout=180000,
    )
    assert page.locator("video").nth(1).evaluate("v=>!!v._hls"), "未使用 hls.js 播放"
    hls_version = page.evaluate("Hls.version")
    page.locator("video").nth(1).evaluate("v=>{v.muted=true;return v.play()}")
    page.wait_for_function(
        "() => document.querySelectorAll('video')[1].currentTime>1", timeout=90000
    )
    duration = page.locator("video").evaluate_all("vs=>vs.map(v=>v.duration)")
    assert abs(duration[0] - duration[1]) < 0.1
    page.locator("video").nth(1).evaluate("v=>v.currentTime=121")
    page.wait_for_function(
        "() => document.querySelectorAll('video')[1].currentTime>121.2 && !document.querySelectorAll('video')[1].seeking",
        timeout=90000,
    )
    with page.expect_download(timeout=300000) as pending:
        page.get_by_role("link", name="下载成品", exact=True).click()
    pending.value.save_as(str(out / "result.mp4"))
    with page.expect_download(timeout=300000) as pending:
        page.get_by_role("button", name="下载 HLS 打包文件").click()
    pending.value.save_as(str(out / "hls.zip"))
    import hashlib, zipfile

    segment_hashes = page.evaluate(
        r"""async id=>{
      const base='/v1/jobs/'+id+'/hls/';
      const manifest=await(await api(base+'index.m3u8')).text();
      const names=manifest.split(/\r?\n/).filter(s=>s&&!s.startsWith('#'));
      const items=[];
      for(const name of names){const bytes=await(await api(base+name)).arrayBuffer();const digest=await crypto.subtle.digest('SHA-256',bytes);items.push({name,sha256:[...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,'0')).join('')})}
      return items;
    }""",
        job,
    )
    with zipfile.ZipFile(out / "hls.zip") as archive:
        for item in segment_hashes:
            assert (
                hashlib.sha256(archive.read(item["name"])).hexdigest() == item["sha256"]
            )
    (out / "browser-segments.json").write_text(json.dumps(segment_hashes, indent=2))
    page.screenshot(path=str(out / "console-playback.png"), full_page=True)
    (out / "browser.json").write_text(
        json.dumps(
            {
                "job": job,
                "durations": duration,
                "hls_playback": True,
                "hls_version": hls_version,
                "seek": True,
                "mp4_download": True,
                "zip_download": True,
                "verified_remote_segments": len(segment_hashes),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("HLS 播放、跳转、时长对比和两种下载通过", flush=True)
    page.get_by_role("button", name="退出登录", exact=True).click()
    assert page.locator("#login").is_visible() and page.locator("video").count() == 0
    assert page.evaluate("sessionStorage.getItem('consoleToken')") is None
    print("退出登录与媒体清理通过", flush=True)
    browser.close()
