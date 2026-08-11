# Agent 接入说明

## 当前能做什么

Agent 只负责把一条文本命令转换为 `TransferCommand`。`agent-plan` 会复用现有视觉和运动规划流程打印全部航点，但不会初始化夹爪，也不会发送机械臂运动。`agent-transfer` 会在同样的确定性校验之后，复用现有 `transfer` 的真机安全门禁、执行前复扫和抓放流程。

```text
文本 → CommandAgent → TransferCommand → 确定性校验
     → 现有 plan-transfer → 视觉扫描与运动规划 → 打印
```

跨架命令可以被 Gemini 理解，但在导航实现前仍由现有同架安全锁拒绝。即使使用
`agent-transfer`，Gemini 也只生成结构化命令；Python 项目仍负责所有校验和运动执行。
真机模式下先确认空载并输入 `EMPTY`，程序自动进入观察位；预览后还必须输入 `MOVE`
才会开始抓放。携管目标复检、最新坐标重规划和最终闭环复扫不能被 Agent 绕过。

## 离线模式

默认配置为：

```yaml
agent:
  provider: local
  model: "gemini-3.5-flash"
  api_key_env: "GEMINI_API_KEY"
```

`local` 不调用网络，只从文本中读取恰好两个标准槽位地址。第一个是源，第二个是目标：

```powershell
python -m tube_grabber agent-plan --text "rack_1.r1c1 -> rack_1.r1c2"
```

少于或多于两个地址时只要求补充信息，不生成计划。

## Gemini 模式

1. 从 Google AI Studio 创建自己的 API key，绝不能复制其他仓库里的密钥。
2. 把密钥保存为操作系统环境变量 `GEMINI_API_KEY`，不要写入 YAML、Python、日志或 Git。
3. 安装可选依赖：

```powershell
python -m pip install -e ".[agent]"
```

4. 保持 `config/app.yaml` 默认的 `local`，使用命令行参数只为本次测试切换到 Gemini。
5. 先运行：

```powershell
python -m tube_grabber doctor
python -m tube_grabber agent-plan --agent-provider gemini --text "把一号架第一行第一列的试管移动到一号架第一行第二列"
```

确认模型理解、视觉结果和规划输出全部正确后，才运行真实执行入口：

```powershell
python -m tube_grabber agent-transfer --agent-provider gemini --text "把一号架第一行第一列的试管移动到一号架第一行第二列"
```

在 `runtime.mode: fake` 下，该命令只驱动假硬件。切换到 `real` 后，它才会连接真实设备，
并继续受到运动参数确认、自动观察位、同架限制、`EMPTY`/`MOVE` 授权、执行前复扫、携管
目标复检和最终闭环验证保护。

Gemini 不会获得任何机械臂函数。它只能提出 `propose_transfer` 的六个参数；自动函数调用被关闭，Python 再检查：

- 必须恰好一个函数调用；
- 必须恰好六个字段，不能多也不能少；
- rack 只能是 `rack_1` 或 `rack_2`；
- 行列必须是整数；
- 行只能是 1～2，列只能是 1～6；
- 源和目标不能相同。

API不可用、模型输出不完整或参数非法时，命令以错误或澄清结束，不会回退为猜测。

## 代码入口

| 文件 | 职责 |
|---|---|
| `agent/models.py` | `AgentDecision` 和最小 `CommandAgent` 接口 |
| `agent/local.py` | 无网络的标准地址解析器 |
| `agent/gemini.py` | Gemini 手动函数调用和不可信输出校验 |
| `agent/prompt.py` | 简短、不可越权的系统指令 |
| `agent/factory.py` | 根据 YAML 选择实现 |
| `cli.py` | `agent-plan` 和 `agent-transfer` 操作者入口 |

未来接入语音时，只需把 ASR 文本传给 `CommandAgent.interpret()`；不要在 `speech/` 中复制任务解析或运动逻辑。
