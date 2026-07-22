普通 `dit` 支持用上一段结果的最后若干帧续写。第一次生成 60 帧：

```bash
python infer.py \
  --model_path save/ddpm263bert_clip_refiner_dit/model000600000.pt \
  --text_prompt "a person raises both hands" \
  --num_frames 60 \
  --out outputs/segment_01.npy \
  --mp4_dir outputs/segment_01_mp4
```

第二次生成 80 个新帧，并把第一段最后 20 帧作为固定前缀：

```bash
python infer.py \
    --model_path save/ddpm263bert_clip_refiner_dit/model000600000.pt \
    --text_prompt "a person claps happily" \
    --num_frames 60 \
    --continuation_npy outputs/segment_01.npy \
    --continuation_prefix_frames 20 \
    --continuation_blend_frames 12 \
    --out outputs/segment_02.npy \
    --mp4_dir outputs/segment_02_mp4
```

服务端支持按 session_id 保存上一段动作，避免不同客户端串线：

```bash
# 第一次：60帧
curl -X POST http://127.0.0.1:8000/generate \
-H "Content-Type: application/json" \
-d '{
    "text": "a person walks forward",
    "num_frames": 60,
    "session_id": "demo",
    "reset_session": true
}' \
--output segment_01.npy

# 第二次：以上一次最后20帧为前缀，返回80个新帧
curl -X POST http://127.0.0.1:8000/generate \
-H "Content-Type: application/json" \
-d '{
    "text": "a person raises both hands",
    "num_frames": 80,
    "session_id": "demo",
    "prefix_frames": 20,
    "blend_frames": 12
}' \
--output segment_02.npy
```

说明：

- `continuation_prefix_frames=20`：扩散时固定上一段最后 20 帧。
- `continuation_blend_frames=12` / `blend_frames=12`：在新段开头用 12 帧平滑姿态和速度边界。
- 新版 NPY 会保存 `decode_state`，下一段会继承根位置和朝向，避免重新回到原点。
