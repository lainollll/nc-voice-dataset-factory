"""Replace only the official notebook's dataset load/encode cells.

Run in the notebook's training environment. Exports contain a standalone loader,
so the desktop application's dependencies do not need to be installed there.
"""

# raw_ds = load_lfm3_dataset(r"H:\datasets\NC_Voice_Dataset")
# encoded = encode_for_lfm3(raw_ds, processor)
# train_dataset, eval_dataset = encoded["train"], encoded["validation"]
# SYSTEM_PROMPT = "Perform TTS in Swedish using My Voice."
# Keep the notebook's collator and trainer; do not shuffle the splits together.
