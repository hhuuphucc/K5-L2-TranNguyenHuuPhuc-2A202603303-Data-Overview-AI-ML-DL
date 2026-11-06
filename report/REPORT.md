# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468; `class_name`= "cab"; `rank`=1 ; `score`= 0.510915; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh. Lý do là vì trường task trong file JSON ghi rõ là image_classification (phân loại ảnh) và record hoàn toàn không có thông tin về tọa độ hộp giới hạn (bbox) hay đa giác (polygon). Phân loại ảnh là bài toán gán nhãn đại diện cho toàn bộ nội dung của bức ảnh đó, khác với phát hiện vật thể (tìm và đóng khung từng vật riêng lẻ).
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp đi kèm taxonomy/checkpoint. Cụ thể ở đây, danh sách lớp được định nghĩa sẵn bởi bộ từ điển ImageNet-1K (gồm 1000 lớp) mà mô hình đã được huấn luyện (lưu trong checkpoint yolo11n-cls.pt). Mô hình học máy không tự "nghĩ" hay sáng tạo ra tên lớp mới, nó chỉ dự đoán và chọn ra lớp phù hợp nhất từ danh sách cố định đã được cung cấp trong quá trình huấn luyện.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Cần ghi đầy đủ class_id, class_name và taxonomy_name vì: Tên lớp (class_name) dùng để con người dễ đọc hiểu, nhưng dễ bị trùng lặp hoặc hiểu sai ngữ nghĩa giữa các ngữ cảnh khác nhau. class_id là mã định danh duy nhất giúp hệ thống máy tính xử lý và phân biệt chính xác. taxonomy_name xác định rõ hệ thống từ điển hoặc bộ dữ liệu gốc (ví dụ: ImageNet hay COCO), đảm bảo tính nhất quán khi tham chiếu và tránh xung đột khi gộp nhiều nguồn dữ liệu.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Khi ảnh có nhiều vật hoặc chủ đề, guideline cần quy định rõ: Tiêu chí ưu tiên để người gán nhãn chọn ra lớp đại diện duy nhất cho toàn bộ bức ảnh.
6. Vì sao không được lấy `score` của model làm ground truth?
- Không được lấy score của model làm ground truth vì: score chỉ đại diện cho độ tự tin (số "tin") của mô hình đối với dự đoán của nó tại thời điểm đó, hoàn toàn không phản ánh điểm chất lượng của nhãn do người gán. Ground truth phải là sự thật chuẩn xác do con người (annotator) trực tiếp đánh giá và gán theo một bộ quy tắc (guideline) chặt chẽ.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= "person"; `score`= 0.912625; `bbox_xyxy`=[385.33,69.24,498.92,348.92] ; `bbox_width`= 113.58; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên-trái: Bắt đầu tại tọa độ x = 385.33, y = 69.24 (tính từ gốc 0,0 ở góc trên cùng bên trái của bức ảnh).
- Góc dưới-phải: Kết thúc tại tọa độ x = 498.92, y = 348.92.
- Kích thước: Hộp giới hạn này có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Với ngưỡng 0.20: Mô hình giữ lại nhiều dự đoán hơn, tổng cộng có 17 vật thể.
- Với ngưỡng 0.60: Mô hình khắt khe hơn, chỉ giữ lại các kết quả có độ tự tin cao, tổng cộng có 6 vật thể.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ ngưỡng (giữ lại nhiều hộp hơn), mô hình sẽ sinh ra nhiều dự đoán sai (false positives), ví dụ như nhận diện bóng râm, đồ vật không rõ ràng, hoặc vẽ nhiều hộp chồng chéo lên cùng một vật thể. Do đó, người reviewer sẽ phải mất nhiều công sức hơn để duyệt thủ công: xóa bỏ các hộp rác, loại bỏ các hộp bị trùng lặp nhãn và căn chỉnh lại các hộp bị vẽ quá rộng hoặc quá chật so với vật thể thực tế.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Quy tắc: "Hộp giới hạn (bounding box) phải được vẽ khít nhất có thể (tight box) bao quanh toàn bộ phần diện tích đang nhìn thấy được của đối tượng. Không được cắt lẹm vào rìa của đối tượng và phải hạn chế tối đa các điểm ảnh nền (background) thừa lọt vào bên trong hộp."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật thể: (Dựa vào danh sách nhãn) Bạn có thể chọn ví dụ một chiếc bát (bowl) nằm khuất một nửa phía sau chiếc bát khác, hoặc tay người (person) bị cắt mất ở rìa cạnh bức ảnh.
- Cách xử lý: Viết thành quy tắc trong guideline.
- Vì sao: Hiện tượng vật thể bị che khuất (occlusion) hoặc lẹm mép ảnh (truncation) là trường hợp cực kỳ phổ biến trong dữ liệu thực tế. Nếu yêu cầu gán nhãn hỏi Lab Coach (escalation) mỗi khi gặp trường hợp này, quy trình sẽ bị ách tắc hoàn toàn.


## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001; `class_name`= "person"; `score`= 0.899318; số điểm polygon=`polygon_point_count`= 348; 3–5 điểm đầu của `polygon_xy`=[
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ]
]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- So với bounding box (hộp chữ nhật chỉ cho biết vị trí, chiều rộng và chiều cao bao quát), polygon (đa giác) cung cấp thêm thông tin chi tiết về đường viền (biên/contour) thực tế và hình dáng cụ thể của vật thể ở mức độ điểm ảnh (pixel). Polygon giúp tách bạch chính xác phần diện tích thuộc về vật thể khỏi phần nền, điều này đặc biệt quan trọng đối với các vật thể có hình dạng phức tạp, uốn lượn, hoặc bị che khuất không có dạng hình khối vuông vức.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id (ví dụ: kitchen-001) là mã định danh duy nhất (ID) được gán riêng cho một cá thể (instance) cụ thể xuất hiện trong bức ảnh đó.
- "Không phải class ID": Vì trong cùng một ảnh có thể có nhiều người (cùng chung class_id là 0 ứng với nhãn "person"), nhưng mỗi người sẽ có một instance_id riêng biệt (ví dụ: kitchen-001, kitchen-002) để phân biệt người này với người kia.
- "Không phải tracking ID": Tracking ID dùng trong video để theo dõi một vật thể di chuyển qua nhiều khung hình (frame) khác nhau theo thời gian. Ở đây là bài toán xử lý ảnh tĩnh (image segmentation), instance_id chỉ có giá trị phân biệt các cá thể nội bộ bên trong một bức ảnh duy nhất này, không liên kết với các ảnh khác.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Quy tắc: "Đường đa giác (polygon) phải bám sát chính xác vào đường viền nhìn thấy được của đối tượng. Khi vật thể bị che khuất một phần, chỉ vẽ mask cho phần diện tích thực sự lộ ra (không tự suy diễn và vẽ thêm phần bị che lấp). Hạn chế tối đa việc khoanh lẹm vào các điểm ảnh của phần nền (background) hoặc của vật thể lân cận."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Vị trí trên ảnh: Trên ảnh kitchen, ta có thể thấy rõ các chiếc bát (bowl 0.70, bowl 0.74) đang đặt trực tiếp trên mặt bàn (dining table 0.58), tạo ra hiện tượng hai vật thể chạm nhau (đường biên của mặt bàn và đáy bát tiếp xúc sát nhau).
- Cách xử lý: Sử dụng Guideline.
- Vì sao: Hiện tượng các vật thể che khuất (occlusion), chạm vào nhau hoặc có biên mờ nhòe (do tiêu cự, điều kiện ánh sáng) là bản chất tất yếu và xuất hiện liên tục trong dữ liệu ảnh thực tế. Nếu người gán nhãn (annotator) dùng escalation (báo cáo lên Lab Coach) cho mọi trường hợp này, tiến độ dự án sẽ bị đình trệ hoàn toàn.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn văn bản (class name) duy nhất đại diện cho toàn bộ nội dung chính của bức ảnh. | Trên ảnh traffic, mô hình bối rối giữa cab (0.51), minibus (0.16) và police_van (0.08) vì ảnh có nhiều loại xe. | Xem ảnh và chọn ra 1 nhãn duy nhất mô tả đúng nhất chủ đề chính theo Guideline. | Kiểm tra xem nhãn được chọn có tuân thủ Guideline (ví dụ ưu tiên vật thể lớn nhất) hay không. |
| Phát hiện vật thể | Tọa độ của một hình chữ nhật (bounding box) bao quanh vật thể và nhãn lớp (class name) tương ứng. | Trên ảnh kitchen, các bát (bowl) chồng lên nhau khiến việc xác định giới hạn box khó khăn; hoặc mô hình nhận diện nhầm không gian trống là cup. | Vẽ hình chữ nhật khít nhất quanh phần nhìn thấy của vật thể và gán nhãn cho nó. | Kiểm tra xem box có quá rộng, quá chật, lẹm vào nền hoặc dán sai nhãn hay không. |
| Instance segmentation | Một tập hợp các điểm tọa độ tạo thành đa giác (polygon) bám sát đường viền vật thể, kèm nhãn lớp và ID riêng. | Trên ảnh kitchen, mask của bàn (dining table) lẹm vào khoảng trống hoặc dính viền với các vật đặt trên đó; mô hình nhận diện nhầm bóng đen thành người (person). | Dùng công cụ vẽ chính xác từng điểm ảnh trên đường viền thực tế của vật thể để tạo polygon. | Kiểm tra các đường viền có bị lem ra ngoài (false positive) hay thiếu chi tiết (false negative) không, và cách xử lý mép bị che khuất. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Quy tắc bảo vệ dữ liệu: Chỉ sử dụng 3 hình ảnh COCO công khai có sẵn được cấp phép theo khuôn khổ bài thực hành, tuyệt đối không tải lên, sử dụng hoặc chia sẻ hình ảnh cá nhân, dữ liệu riêng tư, hoặc tài sản sở hữu trí tuệ nội bộ.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Xử lý vi phạm phạm vi dữ liệu: Nếu gặp hình ảnh hoặc dữ liệu nằm ngoài phạm vi cho phép (như dữ liệu cá nhân, ảnh bạo lực/nhạy cảm), tôi sẽ ngay lập tức dừng xử lý và báo cáo trực tiếp cho Lab Coach / Giảng viên hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
