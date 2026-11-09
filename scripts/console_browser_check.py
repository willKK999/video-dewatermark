"""中文操作台浏览器验收；令牌只从文件读取，输出不含凭证和签名地址。"""
import argparse
from pathlib import Path
import time

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://175.121.93.64:61591/")
    parser.add_argument("--token-file", default="/home/box/.config/dewatermark/console-token")
    parser.add_argument("--browser", default="/opt/google/chrome/chrome")
    parser.add_argument("--output", default="/workspace/user-videos/console")
    parser.add_argument("--upload", type=Path, help="指定时上传并提交；否则检查已有成功任务")
    parser.add_argument("--self-signed", action="store_true", help="允许当前实例的自签名证书")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser, headless=True, args=["--no-sandbox"])
        context = browser.new_context(ignore_https_errors=args.self_signed, viewport={"width": 1360, "height": 1100}, locale="zh-CN", accept_downloads=True)
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded")
        assert page.locator("#login").is_visible()
        page.locator("#token").fill(Path(args.token_file).read_text().strip())
        page.get_by_role("button", name="登录", exact=True).click()
        page.locator("#workspace").wait_for(state="visible")
        assert page.locator("#token").input_value() == ""
        print("登录通过", flush=True)
        page.locator("#submit").click()
        assert "请先选择" in page.locator("#notice").inner_text()
        page.locator("#dropzone").evaluate("""zone => {
            const transfer = new DataTransfer();
            transfer.items.add(new File(['test'], '拖拽验收.mp4', {type:'video/mp4'}));
            zone.dispatchEvent(new DragEvent('drop', {bubbles:true, dataTransfer:transfer}));
        }""")
        assert page.locator("#chosenFilename").inner_text() == "拖拽验收.mp4"
        page.locator("#file").set_input_files([])
        for stage, index in [("probe", 0), ("detect", 1), ("mask", 2), ("inpaint", 3), ("publish", 4)]:
            assert page.evaluate("""([stage,index]) => {
                const steps=stageProgress({status:'RUNNING',stage});
                return steps.children.length===5 && steps.children[index].getAttribute('aria-current')==='step';
            }""", [stage, index])
        print("未选文件拦截、拖拽选择及五阶段展示通过", flush=True)
        if args.upload:
            page.locator("#file").set_input_files(str(args.upload.resolve()))
            page.locator("#submit").click()
            for _ in range(600):
                if "任务已提交" in page.locator("#uploadStatus").inner_text():
                    break
                if page.locator("#notice").inner_text():
                    raise RuntimeError("上传或提交失败，请查看页面提示")
                page.wait_for_timeout(1000)
            else:
                raise RuntimeError("上传或提交超时")
            print("上传与提交通过", flush=True)
        started = time.monotonic()
        previous = None
        while time.monotonic() - started < 1200:
            badges = page.locator(".badge").all_text_contents()
            if badges != previous:
                print("任务状态：" + "、".join(badges), flush=True)
                previous = badges
            if (badges and badges[0] == "成功") or (not args.upload and "成功" in badges):
                break
            if args.upload and badges and badges[0] in ["失败", "需要复核"]:
                raise RuntimeError("任务未通过处理验收")
            page.wait_for_timeout(4000)
        else:
            raise RuntimeError("等待处理超时")
        success = page.locator("#taskList .task").filter(has_text="成功").first
        success.click()
        page.get_by_role("button", name="播放对比与下载").first.click()
        for _ in range(60):
            if page.locator("video").count() == 2 and page.locator("video").evaluate_all("(vs)=>vs.every(v=>v.readyState>=1)"):
                break
            page.wait_for_timeout(2000)
        else:
            raise RuntimeError("双视频未成功加载")
        assert page.locator(".card:not([hidden]) video").count() == 2
        for video in page.locator("video").all():
            video.evaluate("async (v)=>{v.muted=true; await v.play()}")
        page.wait_for_timeout(1200)
        assert page.locator("video").evaluate_all("(vs)=>vs.every(v=>v.currentTime>0 && !v.paused)")
        for video in page.locator("video").all():
            video.evaluate("(v)=>{v.pause();v.currentTime=2}")
        page.wait_for_timeout(1500)
        with page.expect_download() as event:
            page.get_by_text("下载成品", exact=True).first.click()
        event.value.save_as(str(output / "验收成品.mp4"))
        assert (output / "验收成品.mp4").stat().st_size > 0
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(output / "ui-v2-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(400)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(output / "ui-v2-mobile.png"), full_page=True)
        page.reload(wait_until="domcontentloaded")
        page.locator("#workspace").wait_for(state="visible")
        page.get_by_role("button", name="退出登录").click()
        assert page.locator("#login").is_visible()
        assert page.evaluate("()=>sessionStorage.getItem('consoleToken')") is None
        browser.close()
        print("播放对比、下载、截图、会话恢复及退出清除通过", flush=True)


if __name__ == "__main__":
    main()
