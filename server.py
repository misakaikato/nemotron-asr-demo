"""Nemotron 3.5 ASR Streaming 0.6B 本地体验 demo。

两条路径:
  WS  /ws         — 麦克风 PCM(16k int16) → cache-aware 真流式,逐块回推累计文本+速度指标
  POST /transcribe — 整文件批式转写,回 RTFx

真流式复用 mlx_audio 内部管线:增量 mel(log_mel_spectrogram_frames,全音频等价)
→ stream_encode_chunks(逐层 attention/conv cache,O(n) 不重算)→ 逐块 RNNT 解码。
"""

import asyncio
import queue
import tempfile
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from mlx_audio.stt import load
from mlx_audio.stt.models.nemotron_asr import tokenizer as tok
from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram_frames
from mlx_audio.stt.models.nemotron_asr.nemotron_asr import (
	AlignedToken,
	sentences_to_result,
	tokens_to_sentences,
)
from mlx_audio.stt.models.nemotron_asr.streaming import stream_encode_chunks

# 优先用项目内权重,没有再回落到 HF 下载
_LOCAL_MODEL = Path(__file__).parent / "models" / "nemotron-3.5-asr-streaming-0.6b"
MODEL_ID = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "mlx-community/nemotron-3.5-asr-streaming-0.6b"
SAMPLE_RATE = 16_000
MIN_EMIT_FRAMES = 32  # 攒够 ~0.32s mel 帧才喂编码器,减少小 chunk 开销

print(f"[startup] loading {MODEL_ID} ...", flush=True)
_t0 = time.perf_counter()
model = load(MODEL_ID)
PRE = model.preprocessor_config
print(f"[startup] loaded in {time.perf_counter() - _t0:.1f}s", flush=True)

# 预热(首次前向触发编译/内存分配,不算进用户看到的速度)
_t0 = time.perf_counter()
model.generate(mx.zeros(SAMPLE_RATE, dtype=mx.float32), language="en-US")
print(f"[startup] warmup {time.perf_counter() - _t0:.1f}s", flush=True)

app = FastAPI()

# (token_id, 匹配用文本) 预计算:▁→空格、小写;跳过特殊 token
_PIECES = [
	(i, tok.piece_to_text(p).lower())
	for i, p in enumerate(model.vocabulary)
	if not tok.is_special_piece(p) and tok.piece_to_text(p).strip()
]


class HotwordBiaser:
	"""文本空间的贪心浅融合热词偏置。

	对每个热词的每个字符位置预扫词表,得到"能把匹配推进下去的 (token, 步长)"表;
	解码每步给这些 token 的 logit 加 weight×步长,argmax 后按实际发射 token 推进/淘汰
	匹配状态。blank 不动状态,发射了无关 token 则清空(greedy 无回退,只做前向偏置)。
	"""

	def __init__(self, words: list[str], weight: float):
		self.weight = weight
		self.words = [w.strip().lower() for w in words if w.strip()]
		# tables[wi][pos] = [(token_id, advance_chars), ...]
		self.tables = []
		for w in self.words:
			table = []
			for pos in range(len(w)):
				cands = []
				for pid, ptext in _PIECES:
					t = ptext[1:] if pos == 0 and ptext.startswith(" ") else ptext
					if t and w.startswith(t, pos):
						cands.append((pid, len(t)))
				table.append(cands)
			self.tables.append(table)
		self.active: set[tuple[int, int]] = set()

	def step_candidates(self):
		"""返回 (token_id → 偏置分, 本步参与匹配的状态集)。"""
		states = self.active | {(wi, 0) for wi in range(len(self.words))}
		cand: dict[int, float] = {}
		for wi, pos in states:
			for pid, adv in self.tables[wi][pos]:
				bonus = self.weight * adv
				if bonus > cand.get(pid, 0.0):
					cand[pid] = bonus
		return cand, states

	def advance(self, token_id: int, states):
		new: set[tuple[int, int]] = set()
		for wi, pos in states:
			for pid, adv in self.tables[wi][pos]:
				if pid == token_id and pos + adv < len(self.words[wi]):
					new.add((wi, pos + adv))
		self.active = new


def decode_biased(prompted_chunks, biaser: HotwordBiaser | None):
	"""复刻 model._decode_prompted_chunks 的贪心 RNNT 循环,argmax 前注入热词偏置。"""
	frame_sec = (
		model.encoder_config.subsampling_factor
		* model.preprocessor_config.hop_length
		/ model.preprocessor_config.sample_rate
	)
	last_token = model.blank_id
	decoder_hidden = None
	hypothesis: list[AlignedToken] = []
	global_time = 0

	for prompted in prompted_chunks:
		chunk_len = prompted.shape[1]
		time_idx = 0
		new_symbols = 0
		while time_idx < chunk_len:
			feature = prompted[:, time_idx : time_idx + 1]
			current_token = (
				mx.array([[last_token]], dtype=mx.int32)
				if last_token != model.blank_id
				else None
			)
			decoder_output, (h, c) = model.decoder(current_token, decoder_hidden)
			decoder_output = decoder_output.astype(feature.dtype)
			proposed_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
			joint_output = model.joint(feature, decoder_output)

			if biaser and biaser.words:
				cand, states = biaser.step_candidates()
				if cand:
					logits = np.array(joint_output.astype(mx.float32)).reshape(-1)
					for pid, bonus in cand.items():
						logits[pid] += bonus
					pred_token = int(np.argmax(logits))
				else:
					pred_token = int(mx.argmax(joint_output))
				if pred_token != model.blank_id:
					biaser.advance(pred_token, states)
			else:
				pred_token = int(mx.argmax(joint_output))

			if pred_token != model.blank_id:
				last_token = pred_token
				decoder_hidden = proposed_hidden
				if not tok.is_special_token(last_token, model.vocabulary):
					hypothesis.append(
						AlignedToken(
							last_token,
							start=(global_time + time_idx) * frame_sec,
							duration=frame_sec,
							text=tok.decode([last_token], model.vocabulary),
						)
					)
				new_symbols += 1
				if model.max_symbols is not None and new_symbols >= model.max_symbols:
					time_idx += 1
					new_symbols = 0
			else:
				time_idx += 1
				new_symbols = 0
		global_time += chunk_len
		yield sentences_to_result(tokens_to_sentences(hypothesis))
		mx.clear_cache()


class StreamSession:
	"""一条 websocket 对应一个解码线程,PCM 进队列,累计结果出队列。"""

	def __init__(self, language: str, att_context: list[int], biaser: HotwordBiaser | None = None):
		self.language = None if language == "auto" else language
		self.att_context = att_context
		self.biaser = biaser
		self.pcm_q: queue.Queue = queue.Queue()
		self.out_q: queue.Queue = queue.Queue()
		self.buf = np.zeros(0, dtype=np.float32)
		self.wait_s = 0.0  # 等麦克风数据的时间,从 RTFx 里剔除
		self.thread = threading.Thread(target=self._run, daemon=True)
		self.thread.start()

	def feed(self, pcm_bytes: bytes):
		self.pcm_q.put(np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0)

	def finish(self):
		self.pcm_q.put(None)

	def _mel_chunks(self):
		"""增量产出与全音频逐帧等价的 mel chunk。帧 i 覆盖样本 [i*hop-n_fft/2, i*hop+n_fft/2),
		只发已"稳定"(不会再被后续音频改变)的帧;结束时补齐尾部。"""
		hop, n_fft = PRE.hop_length, PRE.n_fft
		emitted = 0
		while True:
			t = time.perf_counter()
			item = self.pcm_q.get()
			self.wait_s += time.perf_counter() - t
			if item is None:
				break
			self.buf = np.concatenate([self.buf, item])
			stable = max(0, (len(self.buf) - n_fft // 2) // hop + 1)
			if stable - emitted >= MIN_EMIT_FRAMES:
				x = mx.array(self.buf)
				yield log_mel_spectrogram_frames(x, PRE, emitted, stable)
				emitted = stable
		total = len(self.buf) // hop + 1
		if len(self.buf) > hop and total > emitted:
			yield log_mel_spectrogram_frames(mx.array(self.buf), PRE, emitted, total)

	def _run(self):
		t_start = time.perf_counter()
		try:
			prompted = stream_encode_chunks(
				model, self._mel_chunks(),
				self.language or model.default_language,
				att_context_size=self.att_context,
			)
			for result in decode_biased(prompted, self.biaser):
				compute = time.perf_counter() - t_start - self.wait_s
				audio_s = len(self.buf) / SAMPLE_RATE
				self.out_q.put({
					"type": "partial",
					"text": result.text,
					"audio_s": round(audio_s, 2),
					"compute_s": round(compute, 2),
					"rtfx": round(audio_s / compute, 1) if compute > 0.01 else None,
				})
			self.out_q.put({"type": "final"})
		except Exception as e:  # 解码线程崩了要让前端知道
			self.out_q.put({"type": "error", "message": str(e)})
		finally:
			mx.clear_cache()


@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
	await ws.accept()
	start = await ws.receive_json()  # {language, latency, hotwords, hotword_weight}
	att = {"1.12s": [56, 13], "0.56s": [56, 6], "0.32s": [56, 3], "0.08s": [56, 0]}[
		start.get("latency", "0.56s")]
	words = [w for w in (start.get("hotwords") or "").splitlines() if w.strip()]
	biaser = HotwordBiaser(words, float(start.get("hotword_weight", 3.0))) if words else None
	sess = StreamSession(start.get("language", "auto"), att, biaser)

	async def pump_results():
		while True:
			try:
				msg = sess.out_q.get_nowait()
			except queue.Empty:
				await asyncio.sleep(0.05)
				continue
			await ws.send_json(msg)
			if msg["type"] in ("final", "error"):
				return

	pump = asyncio.create_task(pump_results())
	try:
		while True:
			msg = await ws.receive()
			if msg.get("bytes"):
				sess.feed(msg["bytes"])
			elif msg.get("text") == "stop" or msg.get("type") == "websocket.disconnect":
				break
	except WebSocketDisconnect:
		pass
	finally:
		sess.finish()
	try:
		await asyncio.wait_for(pump, timeout=30)
		await ws.close()
	except Exception:
		pump.cancel()


@app.post("/transcribe")
async def transcribe(
	file: UploadFile,
	language: str = "auto",
	hotwords: str = Form(""),
	hotword_weight: float = Form(3.0),
):
	with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "a.wav").suffix, delete=False) as f:
		f.write(await file.read())
		path = f.name
	lang = None if language == "auto" else language
	from mlx_audio.stt.utils import load_audio
	audio = load_audio(path, SAMPLE_RATE)
	Path(path).unlink(missing_ok=True)
	duration = audio.shape[-1] / SAMPLE_RATE
	words = [w for w in hotwords.splitlines() if w.strip()]

	def run():
		if not words:
			return model.generate(audio, language=lang)
		from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram
		mel = log_mel_spectrogram(audio, PRE)
		prompted = stream_encode_chunks(model, [mel], lang or model.default_language)
		result = None
		for result in decode_biased(prompted, HotwordBiaser(words, hotword_weight)):
			pass
		return result or sentences_to_result([])

	t0 = time.perf_counter()
	result = await asyncio.to_thread(run)
	wall = time.perf_counter() - t0
	return {
		"text": result.text,
		"audio_s": round(duration, 2),
		"compute_s": round(wall, 2),
		"rtfx": round(duration / wall, 1),
	}


@app.get("/")
async def index():
	return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=7860)
