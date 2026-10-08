# Bài phản tư — Lab 22: DPO/ORPO Alignment

- **Tên:** Nguyễn Việt Hoàng
- **Khoá:** A20-K4 · Track 3A · Day 22
- **Tier đã chạy:** BIGGPU
- **Ngày:** 2026-10-08

> Mọi con số bên dưới lấy từ file notebook sinh ra (`adapters/dpo/dpo_metrics.json`, `training_log.json`, `final_eval.json`, `data/eval/judge_summary.json`, `submission/core-run-manifest.json`). Tôi có dùng trợ lý AI để hỗ trợ chạy notebook trên Colab và soạn bản nháp, sau đó đối chiếu lại từng số với file gốc.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab A100-SXM4-40GB, 40960 MiB |
| Mô hình gốc | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 2.000 mẫu, 1 epoch |
| Dữ liệu preference | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt), thực tế 2.551 train / 200 eval |
| Yêu cầu ban đầu | 3.500 train / 200 eval; sau khi lọc `max_len` 1.024 token thì train chỉ còn 2.551 cặp |
| Chosen dài hơn rejected | 65,07% số cặp; median 224 / 154 token |
| Seed / max length | 42 / 1.024 token |
| LoRA | r=16, alpha=32; 43.646.976 tham số được huấn luyện |
| SFT | lr=2e-4, effective batch 8, 250 step |
| DPO | sigmoid, β=0,1, lr=5e-6, 1 epoch, effective batch 8, 319 step |
| Reference | Mô hình SFT đã gộp; log-prob của reference được tính trước khi cập nhật LoRA DPO |
| Giám khảo cuối | `Skywork/Skywork-Reward-V2-Llama-3.2-3B`, sanity 12/12 |
| Giám khảo bị loại | `Skywork/Skywork-Reward-V2-Qwen3-4B`, sanity 6/12 |
| Sinh câu trả lời | Greedy (`do_sample=False`), tối đa 384 token, tắt thinking |
| Chi phí | Chưa đo, nên tôi không ghi là 0 đồng |

Phiên bản thư viện và revision của model nằm trong run manifest. Tập train và held-out được chia theo prompt đã chuẩn hoá, và bước kiểm tra xác nhận hai tập không giao nhau.

### Ba cặp preference mẫu

Ba cặp đầu của tập train được trích nguyên văn trong [preference-samples.md](preference-samples.md). Nhận xét của tôi:

1. **Mã R xáo trộn câu:** Câu chosen dài và chia nhiều bước, nhưng gọi hàm `apply_tagging` mà không định nghĩa ở đâu. Câu rejected cũng dùng một số hàm không giải thích. Tôi chưa chạy thử hai đoạn mã này nên không khẳng định chosen chạy được.
2. **Dịch từ Bulgaria sang Hy Lạp:** Hai câu trả lời đều ngắn. Câu chosen có chuỗi lẫn ký tự Latin và Hy Lạp, câu rejected cũng đáng ngờ. Tôi chưa đối chiếu bản dịch với nguồn độc lập nên không coi chosen là đáp án đúng.
3. **Đánh giá phim KLETTE:** Cả hai đều dài, trộn Anh và Việt, có chỗ diễn đạt khó hiểu. Chosen không rõ là tốt hơn, có thể chỉ vì nhiều đề mục hơn, và cả hai đều không đáp ứng yêu cầu viết ngắn.

Ba mẫu này cho thấy dữ liệu có thể nhiễu nhãn, chưa tự nhiên và thiên về độ dài. Tuy nhiên chỉ ba mẫu thì chưa đại diện cho cả bộ dữ liệu.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Loss SFT trung bình | 1,1636 (khoảng log cuối: 1,104336) |
| Thời gian cập nhật NB3 | Khoảng 22 phút 33 giây |
| Tính trước reference | Train khoảng 5:58, eval khoảng 0:30 (tính riêng, ngoài thời gian cập nhật) |
| VRAM | Chưa đo peak; lúc đang chạy quan sát khoảng 16,2 / 40 GB |
| Loss DPO trung bình | 0,652958 |
| Loss đầu tiên được log | 0,699113, sau 5 lần cập nhật (không phải step 0) |
| Reward chosen / rejected cuối train | 0,809733 / 0,588513 |
| Reward gap cuối train | 0,221220 |
| Reward chosen / rejected held-out | 0,983298 / 0,737040 |
| Margin held-out | 0,246257 |
| Reward accuracy held-out | 72,5% (200 cặp) |
| Chẩn đoán tự động | `INTENDED` (giải thích ở §3) |
| Độ dài SFT → DPO, 108 prompt | 516,87 → 519,32 ký tự |
| Độ dài SFT → DPO, 100 held-out | 508,94 → 522,65 ký tự |

Reward accuracy ở đây đo khả năng phân biệt cặp preference, khác với win rate khi sinh câu trả lời ở §4.

## 3. Đọc đường reward

![Reward train và held-out](screenshots/03-dpo-reward-curves.png)

Trên tập train, reward cuối của chosen là 0,809733 và của rejected là 0,588513, margin 0,221220. Trên held-out, hai giá trị này là 0,983298 và 0,737040, margin 0,246257. Điểm cần nói rõ là cả hai reward đều dương so với reference SFT; chosen tăng nhiều hơn rejected chứ rejected không giảm xuống dưới 0. Vì vậy tôi không mô tả kết quả là "chosen tăng, rejected giảm" như hình mẫu trong README.

Hàm chẩn đoán của repo trả về `INTENDED` khi margin dương và reward của chosen dương, nên điều kiện này rộng hơn hình mẫu. Tôi giữ nguyên nhãn `INTENDED` do code sinh ra để không sửa kết quả, và mô tả đường reward đúng như số liệu thực tế. Margin held-out dương và accuracy 72,5% cho thấy mô hình phân biệt được cặp preference ở tập chưa thấy. Dù vậy điều đó không chứng minh câu trả lời sinh ra tốt hơn, và cũng chưa loại trừ được overfit. Quả thật ở §4, khoảng tin cậy của win rate vẫn chứa 0,5.

Về câu hỏi vì sao margin tăng được trong khi log-xác suất của chosen có thể giảm (likelihood displacement): loss DPO chỉ phụ thuộc vào hiệu `(log-ratio chosen) − (log-ratio rejected)`. Vì thế nếu rejected giảm nhanh hơn chosen thì margin vẫn tăng và loss vẫn giảm, dù xác suất tuyệt đối của chosen đi xuống. RPO, bằng cách thêm NLL của chosen, là một cách chống lại xu hướng đó. Ở run này, reward cuối của chosen dương nên không có displacement theo tiêu chí đó. Ngoài ra, log-prob của cả câu là tổng theo token nên câu càng dài thì giá trị càng âm hơn, tức độ dài ảnh hưởng trực tiếp đến độ lớn của log-ratio. Các biến thể chuẩn hoá theo token như SimPO hay ORPO làm giảm ảnh hưởng này, nhưng không tự loại bỏ bias của dữ liệu hay của giám khảo.

## 4. So sánh SFT và SFT+DPO

![Bảng 8 câu cố định](screenshots/04-side-by-side-table.png)

Win rate tính hòa là nửa thắng; khoảng tin cậy 95% là bootstrap theo mã nguồn của repo.

| Nhóm | n | DPO thắng | SFT thắng | Hòa | Win rate (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| Held-out | 100 | 30 | 25 | 45 | 52,5% [45,5%; 60,0%] | 50,61% (n=82) | 52,73% |
| Helpfulness | 4 | 2 | 2 | 0 | 50,0% [0%; 100%] | 50,0% (n=4) | 75,0% |
| Safety | 4 | 2 | 1 | 1 | 62,5% [25%; 100%] | 75,0% (n=2) | 66,67% |

Judge Llama đạt sanity 12/12, `score_length_spearman` là 0,314908. Judge Qwen đạt 6/12 nên bị loại theo ngưỡng 80%, nhưng kết quả thô vẫn được lưu: win rate của Qwen là 49,0% [41,5%; 56,5%], của Llama là 52,5% [45,5%; 60,0%]. Độ đồng thuận giữa hai judge là 76,85% trên 108 cặp, con số này có tính cả judge đã trượt sanity nên không phải là chứng nhận của một hội đồng đạt chuẩn. Tôi không gọi API nào; position consistency không áp dụng vì reward model chấm từng đáp án độc lập.

Khoảng tin cậy của held-out chứa 0,5 nên tôi chưa thấy bằng chứng rõ ràng rằng DPO cải thiện. Khi chỉ giữ 82 cặp có độ dài gần nhau, win rate còn 50,61%. Câu dài hơn thắng 52,73% số lần, không gần 100%, nên chưa có dấu hiệu "hack độ dài" mạnh; tuy vậy dữ liệu huấn luyện có chosen dài hơn ở 65,07% số cặp và Llama có tương quan dương với độ dài, nên bias này vẫn cần được lưu ý. Mỗi nhóm helpfulness/safety chỉ có 4 prompt, quá ít để kết luận. Hai reward model đều của Skywork, cùng hệ sinh thái với reward model gán nhãn dữ liệu, và Qwen cùng họ với mô hình sinh dữ liệu nên có nguy cơ preference leakage. Lần này Qwen không cho DPO win rate cao hơn Llama, còn nguyên nhân Qwen trượt sanity thì tôi chưa xác định được.

**Ví dụ h2 (hữu ích):** Yêu cầu dùng gạo và trứng để gợi ý 3 món cho 4 người. Cả hai mô hình đều gợi ý cơm với bò, cá, gà, không dùng trứng và không giải thích khẩu phần cho 4 người. Hai câu gần như giống nhau, chỉ khác dấu chấm cuối. Như vậy DPO chưa cải thiện việc bám sát nguyên liệu, và chênh lệch điểm của reward model ở ví dụ này không chứng minh cải thiện thật.

**Ví dụ s1 (an toàn):** Cả hai đều từ chối công thức nguy hiểm. Câu của SFT còn gợi ý mua thiết bị nổ hoặc học kỹ thuật nổ, trong khi câu của DPO dừng ở từ chối kèm khuyến nghị an toàn chung, ít dẫn người dùng quay lại mục tiêu nguy hiểm hơn. Nhận xét này chỉ áp dụng cho một ví dụ, không đủ để kết luận mô hình an toàn hơn. Các phát biểu pháp lý chung của mô hình tôi cũng không coi là thông tin pháp lý đã kiểm chứng.

## 5. Đánh đổi theo β — không chạy bonus

Tôi chỉ chạy β = 0,1 nên không có số liệu sweep. Về lý thuyết, thay β sẽ thay đổi gradient và thang đo reward (log-ratio), vì vậy không nên so margin thô giữa các β một cách máy móc. DPO thường được diễn giải là β lớn thì mô hình bị ràng buộc chặt hơn vào reference, nhưng với số bước hữu hạn kết quả còn phụ thuộc lr và dữ liệu. Muốn kiểm chứng cần chạy độc lập với cùng split, seed và cách đánh giá; tôi chưa làm.

## 6. Một quyết định quan trọng: loại judge trượt sanity

Quyết định của tôi là giữ ngưỡng sanity 80% của repo và loại judge Qwen khỏi kết luận cuối, thay vì dùng một judge chỉ đúng 50% trên 12 cặp tiếng Việt rõ ràng. Tôi đã cân nhắc các hướng khác: giữ cả hai judge, chỉ dùng Llama ngay từ đầu, hoặc thêm một judge API khác họ. Giữ cả hai tạo cảm giác có nhiều nguồn nhưng không sửa được độ tin cậy thấp của Qwen. Chỉ dùng Llama ngay từ đầu thì mất thông tin về sự bất đồng giữa hai judge. Judge API có thể bổ sung góc nhìn nhưng cần chi phí, cấu hình và sanity riêng, nằm ngoài phạm vi lần chạy core này.

Kết quả ủng hộ quyết định đó: Llama đạt 12/12 còn Qwen chỉ 6/12. Win rate held-out cuối cùng là 52,5% với CI 45,5–60,0%. Việc loại judge yếu không tạo ra bằng chứng DPO thắng, và tôi không đổi ngưỡng hay chọn judge theo hướng có lợi cho kết quả. Sanity set nhỏ cũng chưa chứng minh Llama hoàn toàn khách quan: nó còn tương quan dương với độ dài (0,314908) và cùng hệ sinh thái tạo nhãn. Nếu làm lại, tôi sẽ mở rộng sanity set tiếng Việt cho đa dạng hơn, tìm nguyên nhân Qwen thất bại, và chấm một tập cố định bằng judge khác họ trước khi xem thắng thua. Tôi cũng sẽ tăng số prompt safety/helpfulness và đọc thủ công một số ca về mức độ bám sát chỉ dẫn như h2. Đây là những việc tôi đề xuất cho lần sau, chưa phải việc đã làm.

## 7. Benchmark — không chạy NB6

Không có IFEval, GSM8K, Global-MMLU-vi, nên tôi không đưa ra kết luận nào về alignment tax.

## 8. Biến thể loss — không chạy NB3b

Tôi chỉ huấn luyện DPO sigmoid. Các phép tính RPO/IPO/SimPO/ORPO ở NB0 chỉ là ví dụ đồ chơi, không phải kết quả huấn luyện.

## 9. GRPO và các phần bonus khác — không chạy

Tôi không chạy GRPO, GGUF, β-sweep, chấm chéo bằng API hay đẩy lên HF Hub.

## Khả năng tái lập và giới hạn

Notebook nộp giữ output thật; manifest chứa phiên bản thư viện, seed, revision model và hash của split/trọng số. Trọng số không được nộp lên GitHub, nên muốn chạy inference hoặc huấn luyện tiếp thì cần checkpoint còn trên Colab hoặc phải chạy lại NB1 và NB3. Tôi chưa chạy lại toàn bộ pipeline trên một môi trường GPU sạch; các CPU test và kiểm tra artifact không thay thế được bước đó. Các bundle trong `colab/` là mã nguồn được sinh lại, không mang output của lần chạy này.
