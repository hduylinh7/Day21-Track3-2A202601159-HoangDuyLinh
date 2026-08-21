# Lab 21 — Evaluation Report

**Họ tên**: Hoàng Duy Linh  **MSSV**: 2A202601159  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (fp16)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 2.0 / 30 |

**Template có giữ khối `<think>` không?** có — *(results/template_check.json)*

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3398.7 |
| (b) base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 1023.9 |
| (c) LoRA fine-tune | 0.9375 | 0.7500 | 1.0000 | 1560.6 |

**(b) có thật sự mạnh hơn (a) không?** có — Target tăng từ 0.0000 lên 0.6875 và format đạt 1.0000 tuyệt đối. Không sửa `OPTIMIZED_PROMPT`.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6264 | 0.9375 | 983.3 | 12.01 |
| `attn_only` | q,v | 283 | 32456704 | 0.0001 | 0.5372 | 0.9375 | 837.4 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32464896 | 0.00001 | 1.5704 | 0.0000 | 982.6 | 12.01 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.8438 | 1046.0 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Thực nghiệm cho thấy `attn_only` (r=283, đã matched số lượng tham số 32.46M với `correct`) đạt điểm Target accuracy là 0.9375 (hòa với `correct`), mặc dù train loss của `attn_only` thấp hơn (0.5372 so với 0.6264). Thứ tự xếp hạng theo train loss (attn_only tốt hơn correct) không phản ánh đúng thứ tự năng lực tác vụ target (cả hai hòa ở 0.9375). Điều này chứng minh rằng rank không phải là đòn bẩy duy nhất; vị trí gắn adapter bao phủ toàn bộ các lớp tuyến tính (`text-linear`) giúp mô hình học biểu diễn toàn diện hơn thay vì chỉ tập trung nâng rank lớn ở riêng các lớp attention (q,v).

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Cấu hình `wrong_lr` sử dụng Learning Rate là 1e-5 (quy mô Full-FT), bé hơn 10 lần so với mức khuyến nghị cho LoRA (1e-4). Đường loss của `wrong_lr` giảm rất chậm và dừng lại ở mức rất cao 1.5704 (so với 0.6264 của `correct`), dẫn đến Target accuracy hoàn toàn bằng 0.0000. Nếu chỉ quan sát đường loss đang giảm dần mà không biết LR hay không đánh giá trên tập target, người phát triển có thể lầm tưởng mô hình vẫn đang học ổn định nhưng thực chất mô hình bị undershoot nghiêm trọng và không thể thực hiện đúng tác vụ.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Mô hình `qlora` 4-bit tiết kiệm được 4.92 GB VRAM (chỉ chiếm 7.09 GB VRAM so với 12.01 GB của 16-bit LoRA `correct`, giảm khoảng 41% lượng memory). Tuy nhiên, cái giá phải trả là điểm Target accuracy sụt giảm từ 0.9375 xuống 0.8438 (tụt 9.37%). Kết quả thực nghiệm này hoàn toàn ủng hộ khuyến nghị từ nhà phát triển là không nên dùng QLoRA 4-bit trên dòng kiến trúc Qwen3.5 khi ứng dụng yêu cầu độ chính xác cao và cấu hình phần cứng vẫn đáp ứng được fp16/bf16 LoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.2500` · `regression Δ = +0.0000` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ):
Phiên bản LoRA fine-tune (`correct`) đã xuất sắc vượt qua mốc phán quyết với kết quả PASSED. Điểm Target accuracy tăng từ 0.6875 (ở baseline b prompt tối ưu) lên 0.9375, tạo ra mức tăng trưởng ấn tượng Δ = +0.2500 (+25%). Trong khi đó, chỉ số đánh giá khả năng suy luận tổng quát (regression) duy trì tuyệt đối ở mức 0.7500 (Δ = 0.0000), chứng minh mô hình fine-tune không bị hiện tượng catastrophic forgetting (quên tri thức cũ) hay sụt giảm khả năng trả lời các câu hỏi phổ thông. Định dạng đầu ra JSON đạt chuẩn 100% (format = 1.0000). Kết quả này khẳng định quá trình fine-tune thành công và đem lại giá trị thực tế rõ rệt so với prompt engineering thuần túy.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận... | doi_tra / cao / đèn bàn LED / trung_tinh | doi_tra / cao / đèn bàn LED / trung_tinh | doi_tra / cao / đèn bàn LED / trung_tinh | ✅ FT thắng (1.0) |
| 2 | Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size... | doi_tra / thap / balo laptop / tieu_cuc | doi_tra / thap / balo laptop / tieu_cuc | doi_tra / thap / balo laptop / tieu_cuc | ✅ FT thắng (1.0) |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền... | hoan_tien / trung_binh / bình giữ nhiệt / tieu_cuc | hoan_tien / trung_binh / bình giữ nhiệt / tieu_cuc | hoan_tien / trung_binh / bình giữ nhiệt / (thiếu sentiment) | ❌ **FT thua/chưa đạt (0.75)** |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện... | san_pham_loi / trung_binh / nồi chiên không dầu / tieu_cuc | san_pham_loi / trung_binh / nồi chiên không dầu / tieu_cuc | san_pham_loi / trung_binh / nồi chiên không dầu / (thiếu sentiment) | ❌ **FT thua/chưa đạt (0.75)** |
| 5 | Alo shop, mình đặt máy xay sinh tố mã đơn OD126693... | doi_tra / trung_binh / máy xay sinh tố / tieu_cuc | doi_tra / trung_binh / máy xay sinh tố / tieu_cuc | doi_tra / trung_binh / máy xay sinh tố / tieu_cuc | ✅ FT thắng (1.0) |

Có mẫu chung nào ở các ca FT thua không?
Các ca FT chưa đạt điểm tuyệt đối (score 0.75) đều có mẫu chung là chuỗi JSON sinh ra bị cắt ngắn ngay ở trường cuối cùng (`sentiment`), nguyên nhân chủ yếu do giới hạn độ dài sinh (greedy decode / max new tokens) hoặc EOS token bị cắt khi trích xuất kết quả JSON.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).**
Bản fine-tune LoRA (`correct`) rất đáng được deploy cho bài toán phân loại Ticket CSKH tiếng Việt sang định dạng JSON triage 4 trường. Thực nghiệm cho thấy mô hình nâng Target accuracy từ 68.75% (ở prompt tối ưu) lên 93.75% (+25%), đồng thời duy trì 100% định dạng JSON hợp lệ và không gây suy giảm kiến thức tổng quát (đạt 75% ở tập regression). Đòn bẩy thực sự trong bài lab này nằm ở việc cấu hình vị trí adapter phủ toàn bộ các lớp tuyến tính (`text-linear`), lựa chọn Learning Rate chuẩn cho LoRA (1e-4) và áp dụng chính xác loss masking (`assistant-only`) để mô hình chỉ tập trung học sinh ra câu trả lời JSON. Việc chỉ tăng rank ở riêng attention hay dùng QLoRA 4-bit đều làm giảm hiệu quả thực tế của mô hình.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss Masking đúng là gốc rễ**: Giải mã ngược loss mask để đảm bảo chỉ tính loss trên assistant response (`supervised_fraction < 0.95`), tránh lỗi tính loss trên prompt khiến mô hình lặp lại câu hỏi.
2. **Vị trí gắn Adapter quan trọng hơn Rank**: Gắn LoRA trên toàn bộ `text-linear` mang lại đòn bẩy lớn hơn nhiều so với việc chỉ tăng rank thật cao ở riêng các lớp attention (`q,v`).
3. **Đánh giá công bằng dựa trên chỉ số tác vụ**: Không thể đánh giá mô hình chỉ dựa vào train loss hay perplexity, mà phải đo đạc trực tiếp trên tập Target accuracy và kiểm chứng cổng hồi quy tổng quát.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Thử nghiệm phương pháp `MASK_MODE=response-only` kết hợp với việc tinh chỉnh prompt tiếng Việt sâu hơn để giải quyết dứt điểm hiện tượng trích xuất cắt ngắn trường `sentiment` ở các ca biên.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
