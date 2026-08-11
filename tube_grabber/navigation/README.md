# 导航接口预留

当前刻意不实现导航。

以后这里只加入一个最小适配器：

```text
rack_id -> saved chassis station -> wait until the base is stationary
```

这里不得包含视觉或机械臂运动代码。跨站协调器还必须复用应用层的控制器状态、现场复扫和运动确认门，不能直接调用 workflow 绕过门禁。

Woosh ROS2 接口、HOME/RACK_1/RACK_2 建点、跨架状态机、带管运输位、故障恢复和分阶段
验收的完整方案见 [`docs/NAVIGATION_AND_VOICE.md`](../../docs/NAVIGATION_AND_VOICE.md)。
