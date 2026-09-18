const button = document.querySelector("#connect");
const interrupt = document.querySelector("#interrupt");
const status = document.querySelector("#status");
const transcript = document.querySelector("#transcript");
let active = null;

function entry(role, text) {
  document.querySelector("#empty")?.remove();
  const item = document.createElement("article");
  item.className = `entry ${role}`;
  const label = document.createElement("label");
  label.textContent = role === "user" ? "YOU / 你" : "MYBOT / 回复";
  const content = document.createElement("p");
  content.textContent = text;
  item.append(label, content);
  transcript.append(item);
  transcript.scrollTop = transcript.scrollHeight;
}

async function disconnect(message = "麦克风已关闭") {
  const previous = active;
  active = null;
  button.disabled = true;
  if (previous) {
    previous.media?.getTracks().forEach((track) => track.stop());
    previous.socket?.close();
    previous.node?.disconnect();
    if (previous.context && previous.context.state !== "closed") {
      await previous.context.close().catch(() => {});
    }
  }
  button.disabled = false;
  button.textContent = "开启麦克风 ↗";
  interrupt.disabled = true;
  status.textContent = message;
  document.querySelector("#session-state").textContent = "未连接";
  document.querySelector("#level").style.transform = "scale(1)";
}

async function connect() {
  button.disabled = true;
  status.textContent = "正在连接麦克风…";
  const state = {};
  active = state;
  try {
    if (!navigator.mediaDevices)
      throw new Error("请使用本机 Chrome 或 Edge 打开此页面。");
    state.context = new AudioContext({ latencyHint: "interactive" });
    await state.context.resume();
    await state.context.audioWorklet.addModule("/static/audio-worklet.js");
    state.media = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    if (active !== state) {
      state.media.getTracks().forEach((track) => track.stop());
      return;
    }
    state.node = new AudioWorkletNode(state.context, "mybot-audio", {
      outputChannelCount: [1],
    });
    state.node.onprocessorerror = () => {
      if (active === state) disconnect("音频处理异常，请重新开启麦克风。");
    };
    const source = state.context.createMediaStreamSource(state.media);
    const lowpass = state.context.createBiquadFilter();
    lowpass.type = "lowpass";
    lowpass.frequency.value = 7000;
    source
      .connect(lowpass)
      .connect(state.node)
      .connect(state.context.destination);
    const socket = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/audio`,
    );
    state.socket = socket;
    socket.binaryType = "arraybuffer";
    state.node.port.onmessage = ({ data }) => {
      if (socket.readyState !== WebSocket.OPEN) return;
      if (data.type === "mic") {
        if (socket.bufferedAmount > 64000) {
          disconnect("连接积压，请重新开启麦克风。");
          return;
        }
        socket.send(data.pcm);
        const samples = new Int16Array(data.pcm);
        let peak = 0;
        for (const value of samples)
          peak = Math.max(peak, Math.abs(value) / 32768);
        document.querySelector("#level").style.transform =
          `scale(${1 + Math.min(1.1, peak * 5)})`;
      } else if (data.type === "played") socket.send(JSON.stringify(data));
      else if (data.type === "error") disconnect(data.value);
    };
    socket.onmessage = ({ data }) => {
      if (data instanceof ArrayBuffer) {
        const generation = new DataView(data).getUint32(0, true);
        state.node.port.postMessage(
          { type: "audio", generation, samples: new Int16Array(data, 4) },
          [data],
        );
        return;
      }
      const message = JSON.parse(data);
      if (message.type === "clear")
        state.node.port.postMessage({
          type: "clear",
          generation: message.value,
        });
      else if (message.type === "connected") {
        button.disabled = false;
        button.textContent = "关闭麦克风";
        interrupt.disabled = false;
        status.textContent = "正在聆听，说完后稍作停顿即可。";
        document.querySelector("#session-state").textContent = message.value
          .barge_in
          ? "可语音打断"
          : "轮流对话";
      } else if (message.type === "user" || message.type === "assistant") {
        entry(message.type, message.value);
        status.textContent =
          message.type === "user" ? "正在生成并播放回复…" : "正在播放回复…";
      } else if (message.type === "metrics") {
        const m = message.value;
        for (const [id, value] of [
          ["asr", m.asr_ms],
          ["tts", m.tts_ttfa_ms],
          ["first-audio", m.playback_first_audio_ms],
        ]) {
          document.getElementById(id).textContent =
            value == null ? "-" : `${Math.round(value)} ms`;
        }
        status.textContent = "正在聆听，可以继续说话。";
      } else if (message.type === "error") status.textContent = message.value;
    };
    socket.onclose = () => {
      if (active === state) disconnect("连接已断开，请重新开启麦克风。");
    };
    socket.onerror = () => {
      if (active === state)
        disconnect("无法连接服务，请确认后台任务仍在运行。");
    };
  } catch (error) {
    await disconnect(
      error.name === "NotAllowedError"
        ? "麦克风权限未开启，请在地址栏允许访问后重试。"
        : error.message,
    );
  }
}
button.addEventListener("click", () => (active ? disconnect() : connect()));
interrupt.addEventListener("click", () => {
  if (active?.socket?.readyState === WebSocket.OPEN) {
    active.socket.send(JSON.stringify({ type: "interrupt" }));
    status.textContent = "已打断回复，正在聆听。";
  }
});
window.addEventListener("pagehide", () => disconnect());
