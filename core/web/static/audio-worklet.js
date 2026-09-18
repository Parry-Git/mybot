class MyBotAudio extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = new Float32Array(24000 * 4);
    this.read = 0;
    this.write = 0;
    this.count = 0;
    this.phase = 0;
    this.generation = 0;
    this.played = 0;
    this.input = new Int16Array(320);
    this.inputPosition = 0;
    this.inputPhase = 0;
    this.inputSum = 0;
    this.inputCount = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "clear") {
        this.read = this.write = this.count = this.phase = this.played = 0;
        this.generation = data.generation;
      } else if (data.type === "audio" && data.generation === this.generation) {
        if (this.count + data.samples.length > this.ring.length) {
          this.port.postMessage({
            type: "error",
            value: "播放缓冲区溢出，请重新连接。",
          });
          return;
        }
        for (const value of data.samples) {
          this.ring[this.write] = value / 32768;
          this.write = (this.write + 1) % this.ring.length;
        }
        this.count += data.samples.length;
      }
    };
  }
  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    if (input) {
      for (const value of input) {
        this.inputSum += value;
        this.inputCount++;
        this.inputPhase += 16000;
        if (this.inputPhase >= sampleRate) {
          this.inputPhase -= sampleRate;
          const averaged = this.inputSum / this.inputCount;
          this.inputSum = this.inputCount = 0;
          this.input[this.inputPosition++] = Math.round(
            Math.max(-1, Math.min(1, averaged)) * 32767,
          );
          if (this.inputPosition === 320) {
            this.port.postMessage({ type: "mic", pcm: this.input.buffer }, [
              this.input.buffer,
            ]);
            this.input = new Int16Array(320);
            this.inputPosition = 0;
          }
        }
      }
    }
    const output = outputs[0][0];
    for (let i = 0; i < output.length; i++) {
      if (!this.count) {
        output[i] = 0;
        continue;
      }
      const next = (this.read + 1) % this.ring.length;
      output[i] =
        this.ring[this.read] * (1 - this.phase) +
        (this.count > 1 ? this.ring[next] : this.ring[this.read]) * this.phase;
      this.phase += 24000 / sampleRate;
      const consumed = Math.min(this.count, Math.floor(this.phase));
      this.phase -= consumed;
      this.read = (this.read + consumed) % this.ring.length;
      this.count -= consumed;
      this.played += consumed;
    }
    if (this.played >= 480 || (!this.count && this.played)) {
      this.port.postMessage({
        type: "played",
        generation: this.generation,
        samples: this.played,
      });
      this.played = 0;
    }
    return true;
  }
}
registerProcessor("mybot-audio", MyBotAudio);
