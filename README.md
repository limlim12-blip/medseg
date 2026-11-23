# Dự án Phân đoạn Y tế

Thử nghiệm các kiến trúc U‑Net (từ đầu vs. encoder ResNet pretrained) cho phân đoạn ảnh y tế nhị phân,
đánh giá bằng Dice và IoU. Dự án dùng `Dataset` thông thường (map-style: `__len__` + `__getitem__`),
nên `DataLoader` tự xáo trộn dữ liệu và một epoch là một lượt duyệt qua toàn bộ tập train.

## Cấu trúc dự án

```text
├── base/                # Lớp cơ sở cho trainer, data loader và model
├── data/                # Thư mục dữ liệu (bị ignore bởi git)
├── logger/              # TensorBoard và cấu hình logging
├── notebooks/           # Notebook thử nghiệm (UNet baseline)
├── saved/               # Checkpoint, log, history.json, kết quả đánh giá
├── utils/               # MetricTracker và các hàm tiện ích
├── config.json          # Cấu hình chính: model, dữ liệu, tối ưu, huấn luyện
├── dataset.py           # MedicalSegmentationDataset, ghép cặp ảnh/mask, chia train/val/test
├── data_loader.py       # MedicalDataLoader (train) + split_validation() / split_test()
├── download_sample_data.py  # Sinh dữ liệu tổng hợp để chạy thử
├── main.py              # Điểm vào huấn luyện
├── Makefile             # Lệnh tắt cho các tác vụ thường dùng
├── model.py             # UNetScratch, UNetPretrained, SegFormer, các loss và metric (Dice / IoU)
├── parse_config.py      # Đọc config, tạo thư mục saved/models|log/<name>/<run_id>
├── pyproject.toml       # Metadata và phụ thuộc
├── sweep.yaml           # Cấu hình sweep của Weights & Biases
├── sam_eval.py          # Đánh giá zero-shot SAM (prompt box / point) trên tập test
├── test.py              # Đánh giá checkpoint trên tập test
├── train_def.py         # Lớp Trainer
└── visualize.py         # Learning curves và so sánh dự đoán
```

## Bắt đầu nhanh

1. Clone repo và cài phụ thuộc:
   ```bash
   uv sync                      # hoặc: pip install -e .
   # tùy chọn: uv sync --extra smp --extra wandb
   ```
2. Chuẩn bị dữ liệu. Cần thư mục có `images/` và `masks/` (mask trùng tên file hoặc có hậu tố `_mask` / `_segmentation`).
   Để chạy thử, sinh dữ liệu tổng hợp:
   ```bash
   make data        # = python download_sample_data.py --dir data/sample_medical
   ```
   Với dữ liệu thật (ví dụ Kvasir‑SEG), đặt `data_loader.data_dir` trong `config.json`.
3. Huấn luyện / đánh giá:
   ```bash
   make train
   make test CKPT=saved/models/medseg/<run_id>/model_best.pth
   make visualize RUN=saved/models/medseg/<run_id>
   ```

### Cấu hình

Mọi tham số nằm trong `config.json`:

    loss:         bce_dice | dice | focal | tversky
    arch:         model_type (scratch | pretrained | segformer), backbone (resnet34 | resnet50),
                  segformer_name (nvidia/mit-b0 ... b5), base_c, ...
    data_loader:  data_dir, batch_size, img_size, validation_split, test_split, num_workers
    optimizer:    AdamW (lr, weight_decay, amsgrad); weight decay không áp dụng cho bias/norm
    lr_scheduler: CosineAnnealingLR (T_max, eta_min) sau giai đoạn warmup tuyến tính
    trainer:      epochs, warmup_epochs, save_period, monitor ("max val_dice_coeff"), early_stop, amp, compile

Có thể ghi đè nhanh từ dòng lệnh: `make train ARGS="--lr 3e-4 --bs 16"`.
Checkpoint lưu trong `saved/models/<name>/<run_id>/` (`model_best.pth` luôn là epoch tốt nhất theo `monitor`).
Huấn luyện tiếp: `make resume CKPT=saved/models/<name>/<run_id>/checkpoint-epoch5.pth`.

### SegFormer (Transformer)

Cần `transformers` (`uv sync --extra transformers`). `pretrained: true` tải encoder MiT pretrained từ Hugging Face
(decoder được khởi tạo mới); `pretrained: false` huấn luyện từ đầu.

```bash
make segformer                          # nvidia/mit-b0
make segformer SEGFORMER=nvidia/mit-b2  # biến thể lớn hơn (b0..b5)
```

### Đánh giá zero-shot với SAM

`sam_eval.py` chạy SAM (Hugging Face) **không fine-tune** trên đúng tập test và độ phân giải như các mô hình khác,
nên Dice / IoU so sánh trực tiếp được. SAM cần prompt, được sinh từ ground-truth: `box` (hộp bao của tổn thương)
và `point` (một điểm trong mask gần tâm nhất). Cần nêu rõ rằng SAM nhận thông tin ground-truth qua prompt.

```bash
make sam ARGS="--data_dir data/kvasir-seg"                        # sam-vit-base, cả box và point
make sam ARGS="--model_name facebook/sam-vit-large --prompt box"
```

Kết quả: `saved/sam_zero_shot_metrics.json` và `saved/figures/sam_zero_shot.png`.

### Weights & Biases sweep

Đặt `use_wandb: true`, cài `wandb`, rồi `wandb sweep sweep.yaml` và `wandb agent <sweep_id>`.
Metric của sweep là `val/dice_coeff`.

### Lệnh Makefile

| Lệnh | Mô tả |
| --- | --- |
| `make help` | Liệt kê các lệnh |
| `make data` | Sinh dữ liệu tổng hợp |
| `make train` | Huấn luyện bằng `main.py` |
| `make resume CKPT=...` | Huấn luyện tiếp từ checkpoint |
| `make segformer` | Huấn luyện SegFormer |
| `make sam` | Đánh giá zero-shot SAM |
| `make test CKPT=...` | Đánh giá checkpoint trên tập test |
| `make visualize RUN=...` | Vẽ learning curves và dự đoán |
| `make format` | Chạy pre-commit |
| `make clean` / `make clean-logs` | Dọn file sinh tự động / log |
# medseg
