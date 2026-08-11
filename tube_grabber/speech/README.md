# 语音接口预留

当前刻意不实现语音。

以后它唯一的输出是 `TransferCommand`，再交给统一的应用层执行入口。语音模块不得直接调用 workflow 或硬件。

本机音频现状、旧 Grabber 可复用部分、离线 KWS/VAD/ASR/TTS 方案、外接电脑方案、确认与
取消规则以及验收步骤见
[`docs/NAVIGATION_AND_VOICE.md`](../../docs/NAVIGATION_AND_VOICE.md)。
