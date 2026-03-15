## 环境配置

```bash
conda create -n motion python=3.10 -y
conda activate motion
```

```bash
pip install -r requirements.txt
```

下载文本编码器权重：

```bash
mkdir -p language_models
hf download distilbert/distilbert-base-uncased  --local-dir ./language_models/distilbert-base-uncased
```

下载模型：

```python
pip install modelscope

modelscope download --model wangyicheng2/motion_gen_models
```

模型 checkpoint 放在 `save/` 目录下

## 当前仓库支持范围

- `trans_dec`
- DiP 风格自回归采样
- DDPM 采样
- HumanML3D 解码与 MP4 可视化

## 单句推理

```bash
python infer.py \
  --model_path save/dip/model000600343.pt \
  --text_prompt "a person waves both hands" \
  --motion_length 6 \
  --out outputs/sample.npy \
  --mp4_dir outputs/sample_mp4
```

默认参数已经对齐到 DiP 推理路径：

- 自动开启自回归（当 checkpoint 里有 `context_len/pred_len` 时）
- `guidance_param=7.5`
- `num_samples=1`
- `num_repetitions=1`
- `dynamic_text_blend=0.35`
- `autoregressive_motion_blend_frames=12`

输出的 `.npy` 是一个 dict：

- `motion`：`[N, J, 3, T]` 的 XYZ 关节序列
- `text`：文本列表
- `lengths`：每个样本的帧数
- `fps`：固定为 `20`

## 动态文本分段

`--dynamic_text_path` 支持“一个动作里按时间切换多段文本”。

- 文本文件每行一句
- 每一行对应一个自回归窗口
- 总时长会自动按 `行数 * pred_len / 20` 计算
- 当前 DiP checkpoint 会自动走自回归，一般不需要再手动传 `--autoregressive`

示例 `asserts/prompts.txt`：

```txt
a person stands still
a person raises both arms
a person waves both hands
a person claps hands
a person claps hands
a person claps hands
a person punches in the air
a person punches in the air
a person punches in the air
a person twists the upper body left and right
a person twists the upper body left and right
a person twists the upper body left and right
a person waves one hand
a person waves one hand
a person waves one hand
a person waves one hand
```

推荐命令：

```bash
python infer.py \
  --model_path save/dip/model000600343.pt \
  --dynamic_text_path asserts/prompts.txt \
  --out outputs/dynamic_blend.npy \
  --mp4_dir outputs/dynamic_blend_mp4
```

## 关键参数

- `--guidance_param`：文本约束强度。大一些更贴文本，小一些通常更顺。
- `--dynamic_text_blend`：窗口切换时，对前后文本条件做软过渡，范围 `0~1`。
- `--autoregressive_motion_blend_frames`：窗口切换时，对前一段末尾和后一段开头做动作过渡的帧数。
- `--motion_length`：单句推理时的时长，单位秒。
- `--mp4_dir`：填写后会额外导出 MP4。

## 启动服务

```bash
bash serve/run_server.sh save/dip/model000600343.pt --port 8000 --device 0
```

一次性返回整段 `.npy`：

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"a person waves both hands"}' \
  --output results.npy
```

服务端也支持多句分段输入，字段名是 `texts`：

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "a person stands still",
      "a person raises both arms",
      "a person waves both hands",
      "a person claps hands"
    ]
  }' \
  --output results.npy
```

## 流式服务

`POST /generate_stream` 会在每个自回归窗口完成后，立即返回一行 `NDJSON`。

- 单句模式：按 `motion_length` 生成，多段返回
- 多句模式：传 `texts`，一条句子对应一段
- 当前 DiP checkpoint 里 `pred_len=40` 且 `fps=20`，默认一段 `2` 秒

流式多句示例：

```bash
curl -N -X POST http://127.0.0.1:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "a person stands still",
      "a person raises both arms",
      "a person waves both hands"
    ]
  }'
```

每一行返回的 JSON 包含：

- `segment_index`
- `num_segments`
- `segment_frames`
- `segment_seconds`
- `frames`
- `seconds`
- `fps`
- `text`
- `motion`
- `hml_vec`
- `robot`
- `done`

说明：

- `motion` 是当前这一段的连续 XYZ 结果
- `hml_vec` 是当前这一段可连续拼接的 HumanML 向量表示
- `robot.root_pos` 和 `robot.root_rot_ang` 更适合直接拿去做控制参考
- 这条流式链路是stateful decode，不是“每段独立 decode 再硬拼”

## 流式客户端

自带一个最小客户端脚本 [serve/test_stream_client.py](/home/wyc/motion-gen/serve/test_stream_client.py)：

- 边收流式结果，边保存每段 `segment_XXX.npy`
- 基于 `hml_vec` 做 stateful decode，保证段间连续
- 最后自动拼一个完整 `results.npy`
- 额外导出 `robot` 字段，便于机器人控制使用

多句测试：

```bash
python3 serve/test_stream_client.py \
  --url http://127.0.0.1:8000/generate_stream \
  --texts_path asserts/prompts.txt \
  --out_dir outputs/stream_client
```

单句测试：

```bash
python3 serve/test_stream_client.py \
  --url http://127.0.0.1:8000/generate_stream \
  --text "a person waves both hands" \
  --motion_length 8 \
  --out_dir outputs/stream_single
```

输出目录里会有：

- `segment_000.npy`, `segment_001.npy`, ...
- `results.npy`

`results.npy` 除了 `motion/text/lengths/fps` 之外，还包含：

- `hml_vec`：完整拼接后的 HumanML 向量序列
- `robot.joints_xyz`：完整 XYZ 关节序列
- `robot.root_pos`：根节点位置轨迹
- `robot.root_rot_ang`：根节点朝向轨迹

## 性能测试

仓库自带一个最小基准脚本 [serve/benchmark_server.py](/home/wyc/motion-gen/serve/benchmark_server.py)，可以连续请求服务端并统计平均值、P50、P95。

默认同时测试：

- `POST /generate` 的整段返回耗时
- `POST /generate_stream` 的首段返回耗时和整段完成耗时

示例：

```bash
python3 serve/benchmark_server.py \
  --base_url http://127.0.0.1:8000 \
  --runs 20
```

只测流式接口：

```bash
python3 serve/benchmark_server.py \
  --base_url http://127.0.0.1:8000 \
  --mode stream \
  --runs 20 \
  --texts_path asserts/prompts.txt
```
