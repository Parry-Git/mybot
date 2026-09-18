# 开发与验收

## 依赖与代码约定

- 依赖、开发工具和 pytest/Ruff 配置统一维护在 `pyproject.toml`，不再创建平行的 requirements 清单。
- Python 3.12，源码检出后 `python -m pip install -e '.[dev,browser]'`；只做单元测试可安装 `.[dev]`。
- GPU 关键依赖固定到实测版本。升级 PyTorch/vLLM/native TTS 时必须重新做真实链路测试，不单独安装独立 `flash-attn`。
- 新模块按职责放入 ASR、LLM、TTS、audio 或 web，不在 `utils` 中堆实验代码；入口只装配，不被业务代码反向导入。
- 改动取消、队列或会话逻辑必须补充 `tests/unit`。模型实验保留结论与复现条件，不保留失效脚本和权重副本。
- 提交前运行 `python -m ruff check .`、`python -m ruff format --check .`、`python -m pytest -q`。
- 网页使用原生 JavaScript/CSS，无 npm 运行时依赖。需要格式化时可用 `npm exec --yes --package=prettier@3.6.2 -- prettier --write 'core/web/static/*'`，不生成另一套包清单。
- 不把真实密钥写入源码、测试或日志。模型在 Hugging Face 缓存，参考音频在 `assets/`，输出在 `artifacts/`。

## 测试分层

`tests/unit/` 不加载模型、不访问云 API，覆盖分帧/VAD、ASR 会话、token 过滤、HTTP 取消、有界队列、历史一致性、进程锁及 WebSocket。pytest 默认只收集这一层。

`tests/integration/` 显式使用真实 GPU 和 API，需要单独运行。不要与后台服务同时加载第二份模型。

```bash
# 真实 VAD/ASR/LLM/TTS，生成中文 fixture，检查取消及恢复
./run.sh stop
python tests/integration/verify_chain.py --llm live

# 正式后台服务 + Chromium 麦克风输入 + AudioWorklet 渲染
python -m playwright install chromium
./run.sh start
python tests/integration/verify_browser.py
```

无密钥时 `verify_chain.py --llm loopback` 只替换 LLM 为本地 SSE，仍使用真实 ASR/TTS，但不能声称云端连通。浏览器测试默认读取 `artifacts/verification/input.wav`，也可 `--input` 指定含“介绍”的单声道 PCM16 中文 WAV。

浏览器验收包含连续对话、播放中打断、重连、取消后恢复及移动布局检查。报告、WAV、截图都写入 `artifacts/verification/`。Chromium 使用虚拟音频设备，实际运行网页音频逻辑；报告明确标记 `physical_speaker_verified=false`，真实麦克风、外放回声和音色必须人工验收。

## 运行与排障

`./run.sh start` 检查端口与真实 LLM，加载并预热模型，最后通过带 PID 的 HTTP 健康检查判定就绪。文件锁防止重复实例；`./run.sh stop` 等待正常清理。日志为 `artifacts/mybot.log`，上一份为 `mybot.previous.log`。

```bash
# 不依赖麦克风的前台调试
python main.py --mode text --no-playback

# 直接设备模式，仅用于可用的 Linux 音频环境
python scripts/check_audio.py
python main.py --mode voice

# 显存或 vLLM 兼容性问题
python main.py --mode voice --asr-backend transformers
```

WSLg 的 RDP/PulseAudio 在本机连续双向流测试中出现过系统阻塞，表现为 `pactl info` 超时和音频日志中的 asyncq overrun。PortAudio 与 libpulse 都不能修复卡死的系统服务，因此默认使用 **Windows Chrome/Edge**，WSL 只负责推理。旧 PulseAudio 虚拟设备压力脚本已移除，不能将直接设备模式视为本机已通过的验收路径。

浏览器只监听 `127.0.0.1`，检查同源 Origin，限制一个标签页；暂不提供远程手机访问或公网部署。需要 HTTPS/鉴权时应单独设计，不能简单改为全网监听。

## Git 与历史清理

模型、录音、`.env`、构建缓存和运行产物均由 `.gitignore` 排除。历史清理前必须备份 Git 元数据和未提交工作区；本轮归档还保留了旧模型及本地 LFS 对象，不在仓库内保存旧历史分支。

历史重写保留开发提交，不是删除 `.git` 重新初始化。使用 git-filter-repo 删除模型路径/权重并脱敏旧密钥，验证后以 **明确旧提交 ID 的 --force-with-lease** 更新远端；远端变化时停止，不使用无条件 force 或 mirror push。协作者应重新 clone，避免把旧历史 merge 回来。

重写历史不等于吊销已泄露密钥，也不等于清除 GitHub 的 LFS 存储、缓存或其他人的 clone。旧密钥应作废；远端 LFS 存储清理由仓库所有者按 [GitHub LFS 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage) 处理，本项目不会自动删除 GitHub 仓库。

参考：[敏感数据清理](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)、[AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor/process)、[Chromium 虚拟音频输出](https://github.com/chromium/chromium/blob/main/media/audio/audio_manager_base.cc)。
