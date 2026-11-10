package control

import "testing"

func TestHLSInput(t *testing.T) {
	for _, s := range []string{"kelin/video/index.m3u8", "https://cdn.example/video.m3u8?signature=hidden", "http://cdn.example/master.m3u8"} {
		if !validHLSInput(s) {
			t.Errorf("应接受 %q", s)
		}
	}
	for _, s := range []string{"../a.m3u8", "kelin/../other/a.m3u8", "kelin/%2e%2e/other/a.m3u8", "/a.m3u8", "file:///a.m3u8", "https://user:pass@cdn.example/a.m3u8", "x.mp4", "a.m3u8\n"} {
		if validHLSInput(s) {
			t.Errorf("应拒绝无效输入")
		}
	}
}
func TestHLSOptions(t *testing.T) {
	o := Options{HLS: true, HLSBandwidth: -1}
	if o.Validate() == nil {
		t.Fatal("不得接受负码率")
	}
	o = Options{HLS: true, Prompts: []Prompt{{}}}
	if o.Validate() == nil {
		t.Fatal("HLS 必须自动检测")
	}
}
func TestHLSAssetNames(t *testing.T) {
	for _, name := range []string{"../source.ts", "index.m3u8/secret", "123.ts", "index.m3u8?x", "source.mp4"} {
		if hlsName.MatchString(name) {
			t.Fatal("不得读取任意对象")
		}
	}
	for _, name := range []string{"index.m3u8", "000001.ts"} {
		if !hlsName.MatchString(name) {
			t.Fatal("应接受成品文件名")
		}
	}
}
