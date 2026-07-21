cd ~/data/motion-gen

python infer.py   --model_path save/ddpm263bert_clip_refiner_dit/model000600000.pt   --input_text /home/cxm2/data/motion-gen/problem_test_again.txt --motion_length 6   --out outputs/batch.npy   --mp4_dir outputs/problem_test_again

python run_dynamic_batch.py \
  --repo-dir /home/cxm2/data/motion-gen \
  --source-md /home/cxm2/data/motion-gen/动作批测结果.md \
  --model-path save/ar_ardit_cfg40_cosine600k_ctx_strongreg_genprefix_mid_steps6/model000550000.pt \
  --output-dir outputs/dynamic_batch_mp4 \
  --segment-seconds 2

/home/cxm2/data/motion-gen/save/ar_ardit_cfg40_cosine600k_ctx_strongreg_genprefix_mid_steps6/args.json 修改其中的"pred_len": 120,从而修改生成动作的时间