# 普通 DiT 固定前缀续写：原理、服务启动与测试

本文只说明普通 `dit` checkpoint 的最新续写方式。`ar_dit` 的分窗自回归和 `/generate_stream` 是另一条链路，不要混用。

## 1. 最新修改采用的方法

旧方式中，每次普通 DiT 推理都是独立采样和独立解码。即使把两段结果直接拼起来，第二段开头也没有被上一段动作约束，而且根位置、根朝向会重新初始化，容易出现姿态跳变、速度突变或回到原点。

最新实现使用三层连续性处理：

| 层次 | 最新处理 | 作用 |
| --- | --- | --- |
| 扩散采样 | 从上一段末尾取 `P` 帧作为 `known_motion`，用 `known_mask` 在每个反向扩散步固定这些帧 | 让模型在真实历史动作条件下生成后续动作，而不是生成后再硬拼 |
| HumanML 边界 | 丢弃采样结果中的固定前缀，仅返回 `N` 个新帧；对新段最前面的 `B` 帧施加余弦衰减修正 | 第一帧对齐上一段末态，并在若干帧内平滑恢复到模型原始采样结果 |
| XYZ 解码 | 使用 `recover_from_ric_stateful`，把上一段结束时的 `root_rot_ang` 和 `root_pos` 传给下一段 | 保持世界坐标中的根位置和朝向连续，避免每段从原点重新开始 |

扩散内部采样长度为：

```text
总采样帧数 = 固定前缀帧数 P + 返回的新帧数 N
```

普通 DiT 最大总长度为 196 帧，因此必须满足：

```text
P + N <= 196
```

例如固定 20 帧、生成 80 个新帧时，扩散内部采样 100 帧，最终输出只包含新的 80 帧。

具体代码位置：

- `diffusion/gaussian_diffusion.py`：支持 `known_motion` / `known_mask`。每个反向扩散时刻先用 `q_sample` 得到该时刻的已知前缀，再覆盖掩码区域；最后一步精确恢复干净前缀。
- `utils/sampler_util.py`：`smooth_continuation_boundary` 对新段开头做边界状态和速度修正，并使用余弦权重逐渐衰减。
- `infer.py`：从上一份 NPY 读取 `hml_vec` 和 `decode_state`，完成固定前缀采样、边界平滑和连续解码。
- `serve/generate_server.py`：按 `session_id` 在内存中保存上一段动作及解码状态，后续请求自动走相同续写流程。

新版 NPY 会保存：

- `motion`：XYZ 关节动作。
- `hml_vec`：HumanML 表示，可作为下一次续写输入。
- `decode_state.root_rot_ang`：当前片段结束时的根朝向状态。
- `decode_state.root_pos`：当前片段结束时的根位置状态。
- `continuation_prefix_frames`：本次实际使用的历史前缀帧数。
- `continuation_blend_frames`：本次实际使用的边界平滑帧数。

旧版结果没有 `decode_state` 时，`infer.py` 会根据完整 `hml_vec` 自动重建状态；但输入文件必须有 `hml_vec`，或者本身就是 `[T, D]` 的原始 HumanML 数组。

## 2. 服务端是否使用了最新方法

是。最新提交修改了 `serve/generate_server.py`，普通 DiT 的 `POST /generate` 已接入固定前缀、边界平滑和 stateful decode。

`serve/run_server.sh` 的命令行形式本身没有变化。脚本把模型路径写入 `MDM_MODEL_PATH` 后启动 `serve.generate_server:app`；服务初始化时会读取 checkpoint 同目录下的 `args.json`。当其中 `arch` 为 `dit` 时，就会启用普通 DiT 链路，不需要额外传 `--arch dit`。

注意：

- 普通 DiT 续写使用 `POST /generate`，不要使用只面向 AR-DiT 的 `POST /generate_stream`。
- 会话动作和解码状态保存在服务进程内，服务重启后会清空。
- 必须使用 `--workers 1`。多个 worker 各有独立内存，同一个 `session_id` 可能落到不同进程，无法保证连续。
- 服务端用锁串行执行生成请求，避免多个请求同时修改会话状态。
- 不同客户端必须使用不同的 `session_id`，避免动作串线。

## 3. 启动最新普通 DiT 服务

在仓库根目录执行：

```bash
cd /home/cxm2/data/motion-gen

MDM_CONTINUATION_PREFIX_FRAMES=20 \
MDM_CONTINUATION_BLEND_FRAMES=12 \
bash serve/run_server.sh \
  save/ddpm263bert_clip_refiner_dit/model000600000.pt \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --device 0
```

环境变量说明：

- `MDM_CONTINUATION_PREFIX_FRAMES`：服务默认使用的历史前缀长度，默认 `20`，必须大于 0。
- `MDM_CONTINUATION_BLEND_FRAMES`：服务默认使用的边界平滑长度，默认 `12`，可以设为 `0` 关闭边界平滑。
- `MDM_DEVICE`：由 `--device` 写入，示例使用 GPU 0。
- `MDM_GUIDANCE_PARAM`：可选，覆盖 checkpoint 或架构默认的 guidance。普通 DiT 未设置时默认是 `2.5`。
- `MDM_USE_EMA`：可选，设为 `true` 或 `false` 可覆盖是否使用 EMA 权重。

如果使用其他普通 DiT 权重，替换模型路径即可，但模型旁边必须存在匹配的 `args.json`，且其中 `arch` 应为 `dit`。

## 4. 测试最新服务端

### 4.1 检查服务是否启动

另开一个终端：

```bash
curl -s http://127.0.0.1:8000/health
```

预期返回：

```json
{"status":"ok"}
```

### 4.2 生成第一段

第一段不需要历史动作。`reset_session=true` 会先清空名为 `dit-demo` 的旧会话状态：

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "a person walks forward",
    "num_frames": 60,
    "session_id": "dit-demo",
    "reset_session": true
  }' \
  --output outputs/server_segment_01.npy
```

第一段返回 60 帧，并把这 60 帧及其结束解码状态保存在服务端的 `dit-demo` 会话中。

### 4.3 使用上一段续写第二段

同一个 `session_id` 默认 `continue_from_previous=true`。下面会取上一段末尾 20 帧作为固定前缀，内部采样 `20 + 80 = 100` 帧，最终只返回 80 个新帧：

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "a person raises both hands",
    "num_frames": 80,
    "session_id": "dit-demo",
    "prefix_frames": 20,
    "blend_frames": 12
  }' \
  --output outputs/server_segment_02.npy
```

请求参数说明：

- `num_frames`：精确指定本次返回的新帧数，优先级高于 `motion_length`。
- `session_id`：会话隔离标识。同一条连续动作必须保持一致，不同动作序列必须使用不同值。
- `continue_from_previous`：默认 `true`。设为 `false` 时，本次独立生成，但不会主动删除服务端已有会话。
- `reset_session`：设为 `true` 时，生成前删除该会话已有动作和解码状态。
- `prefix_frames`：覆盖服务端默认前缀帧数；实际值不会超过上一段已有帧数。
- `blend_frames`：覆盖默认平滑帧数；实际值不会超过本次新生成帧数。设为 `0` 可关闭平滑。

如果没有传 `num_frames`，服务会用 `motion_length * 20` 计算帧数。

### 4.4 检查服务端输出

```bash
python - <<'PY'
import numpy as np

for path in ("outputs/server_segment_01.npy", "outputs/server_segment_02.npy"):
    result = np.load(path, allow_pickle=True).item()
    print(path)
    print("  motion:", result["motion"].shape)
    print("  hml_vec:", result["hml_vec"].shape)
    print("  lengths:", result["lengths"].tolist())
    print("  session_id:", result["session_id"])
    print("  prefix:", result["continuation_prefix_frames"])
    print("  blend:", result["continuation_blend_frames"])
    print("  decode_state:", sorted(result["decode_state"]))
PY
```

关键预期：

- 第一段 `lengths` 为 `[60]`，`continuation_prefix_frames` 为 `0`。
- 第二段 `lengths` 为 `[80]`，`continuation_prefix_frames` 为 `20`，`continuation_blend_frames` 为 `12`。
- 两段均包含 `hml_vec` 和 `decode_state`。

测试独立生成时，可以使用新会话，或者显式关闭续写：

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "a person jumps once",
    "num_frames": 60,
    "session_id": "independent-demo",
    "continue_from_previous": false,
    "reset_session": true
  }' \
  --output outputs/server_independent.npy
```

## 5. 使用 infer.py 测试最新续写

### 5.1 第一段：独立生成 60 帧

```bash
cd /home/cxm2/data/motion-gen

python infer.py \
  --model_path save/ddpm263bert_clip_refiner_dit/model000600000.pt \
  --text_prompt "a person walks forward" \
  --num_frames 60 \
  --out outputs/infer_segment_01.npy \
  --mp4_dir outputs/infer_segment_01_mp4
```

### 5.2 第二段：固定前 20 帧并返回 80 个新帧

```bash
python infer.py \
  --model_path save/ddpm263bert_clip_refiner_dit/model000600000.pt \
  --text_prompt "a person raises both hands" \
  --num_frames 80 \
  --continuation_npy outputs/infer_segment_01.npy \
  --continuation_prefix_frames 20 \
  --continuation_blend_frames 12 \
  --out outputs/infer_segment_02.npy \
  --mp4_dir outputs/infer_segment_02_mp4
```

第二条命令会读取第一段的 `hml_vec` 和 `decode_state`。扩散采样时前 20 帧始终受已知动作约束，采样结束后前缀被丢弃，因此 `infer_segment_02.npy` 和生成的 MP4 都只包含新的 80 帧。

### 5.3 继续生成第三段

第二段输出也包含新的结束状态，可以继续作为下一次输入：

```bash
python infer.py \
  --model_path save/ddpm263bert_clip_refiner_dit/model000600000.pt \
  --text_prompt "a person lowers both hands and walks away" \
  --num_frames 80 \
  --continuation_npy outputs/infer_segment_02.npy \
  --continuation_prefix_frames 20 \
  --continuation_blend_frames 12 \
  --out outputs/infer_segment_03.npy \
  --mp4_dir outputs/infer_segment_03_mp4
```

### 5.4 检查 infer 输出

```bash
python - <<'PY'
import numpy as np

for path in (
    "outputs/infer_segment_01.npy",
    "outputs/infer_segment_02.npy",
    "outputs/infer_segment_03.npy",
):
    result = np.load(path, allow_pickle=True).item()
    print(path)
    print("  motion:", result["motion"].shape)
    print("  hml_vec:", result["hml_vec"].shape)
    print("  lengths:", result["lengths"].tolist())
    print("  prefix:", result["continuation_prefix_frames"])
    print("  blend:", result["continuation_blend_frames"])
    print("  root_pos:", result["decode_state"]["root_pos"].shape)
    print("  root_rot_ang:", result["decode_state"]["root_rot_ang"].shape)
PY
```

预期第一段长度为 60、前缀为 0；第二和第三段长度为 80、前缀为 20、平滑为 12。

## 6. 代码级测试

固定前缀、边界平滑和 stateful decode 都有对应单元测试：

```bash
cd /home/cxm2/data/motion-gen

python -m unittest \
  tests.test_continuation \
  tests.test_diffusion_inpainting

python -m py_compile \
  infer.py \
  serve/generate_server.py \
  diffusion/gaussian_diffusion.py \
  utils/parser_util.py \
  utils/sampler_util.py
```

当前相关测试共 4 个，覆盖：

- 已知前缀在扩散结束时是否被精确保留。
- `known_motion` 与 `known_mask` 是否必须成对传入。
- 新段边界是否匹配上一段状态/速度并平滑衰减。
- 分段 stateful decode 是否与整段一次性 decode 一致。

## 7. 常见问题

- 报错 `continuation_npy must contain hml_vec`：输入不是新版 `infer.py` 结果。重新生成第一段，或提供 `[T, D]` 的 HumanML 数组。
- 报错总帧数超过 196：减小 `num_frames` 或 `continuation_prefix_frames`。
- 第二个服务请求没有使用前缀：确认两个请求的 `session_id` 完全相同，服务中途没有重启，并且使用 `--workers 1`。
- 不希望继续上一段：传 `"continue_from_previous": false`；希望同时清掉旧状态则再传 `"reset_session": true`。
- 边界仍偏硬：适当增加 `blend_frames` / `continuation_blend_frames`，但不能超过新段帧数；也可以设为 `0` 对比关闭平滑后的效果。
- 想使用多句动态文本或流式逐段输出：应启动 `ar_dit` checkpoint 并使用 `/generate_stream`，不属于本文的普通 DiT 固定前缀方案。


## 实验性前瞻尾帧

CLI新增 `--continuation_lookahead_frames`（默认0）与 `--continuation_is_final`。服务新增 `lookahead_frames`、`is_final` 及默认值环境变量 `MDM_CONTINUATION_LOOKAHEAD_FRAMES`。仅普通DiT使用。采样P+N+L不超过196；在平滑、解码和保存会话之前丢弃尾帧，仅返回N帧。末段强制L0。较长前缀不能超过已实际返回历史。默认保持原行为；尚未通过最终动作质量验收。CLI的 `--continuation_repaint_repeats` 仅用于消融，默认1。
