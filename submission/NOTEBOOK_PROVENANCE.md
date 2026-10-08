# Notebook và nguồn gốc kết quả

`Lab22_DPO_BigGPU_executed.ipynb` là bản core có output thật, trích từ Colab
ngày 2026-10-08: 91 cells, 45 cells chứa output. Không chạy GPU lại để đóng gói.
Không gán output vào bundle nguồn được sinh lại trong `colab/`.

Nguyên bản 141 cells được lấy qua chức năng xuất notebook của Colab từ kernel,
nén gzip, rồi chuyển qua link tải hiển thị trong output. Gzip nguyên bản giữ
cục bộ ở `docs/harness/Lab22_DPO_BigGPU.raw.ipynb.gz`, SHA-256:
`5fdc6369d1821e8c29f7dd3e36aa3ed69a1dd53fe42cdd14d20c5d7e7a1ffa82`.

`scripts/package_colab_submission.py` chỉ bỏ section bonus chưa chạy và các cell
đóng gói/link tải/kiểm tra cuối. Code, execution counts và nội dung output của
các cell được giữ nguyên. Thêm cell ghi chú, ID ổn định và loại trường metadata
riêng Colab trên output stream để notebook đạt schema Jupyter; không đổi text.
Danh sách cell bị bỏ và hash notebook nộp nằm trong `colab-provenance.json`.

File → Download .ipynb không trả file trong phiên tự động. Lần xuất đầu với các
link ZIP base64 lớn bị treo; đã ngắt và kết nối lại cùng runtime, giữ checkpoint.
Sau khi bỏ riêng output link ZIP đã lưu, xuất notebook nén thành công. Link ZIP
không phải output huấn luyện; mọi log, biểu đồ và kết quả NB0–NB4 vẫn được giữ.
Hộp thoại lưu GitHub đã huỷ; chưa commit/push theo yêu cầu review của chủ bài.

`core-run-manifest.json` giữ bản manifest của gói bằng chứng cuối, gồm package
versions, model revisions, seed, split/trọng số hashes và addendum sửa layout.
`colab-provenance.json` ghi phép chuyển metadata `models/sft-merged/` bổ sung
và mapping `/content/lab22` sang thư mục bài nộp. Adapter config gốc không sửa.

`verify.py` chấp nhận reference Colab đã di chuyển chỉ khi hash manifest,
adapter configs, split, Parquets và metadata model khớp. Mất provenance hoặc
thay đổi dữ liệu vẫn bị từ chối. Điều này xác nhận bằng chứng, không xác nhận
có trọng số để inference trên máy Windows. Trọng số được loại theo README.

Lần chạy này hoàn thành core NB0–NB4; bonus không chạy. CI win rate chứa 0,5,
nên chưa chứng minh DPO tốt hơn SFT. Chỉ Llama qua sanity được dùng kết luận.
Chưa chạy lại toàn bộ trên GPU runtime sạch; kiểm tra CPU/schema/hash không
thay thế điều kiện tái lập đó. Reflection là bản thảo chờ chủ bài xác nhận.
