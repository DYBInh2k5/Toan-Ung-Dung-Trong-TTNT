# Toan-Ung-Dung-Trong-TTNT
Toán Ứng Dụng Trong TTNT


Tổng quan về Học máy (Machine Learning Overview)
1. Định nghĩa và Trụ cột cốt lõi
Học máy (Machine Learning) là một ngành khoa học nghiên cứu các thuật toán cho phép máy tính có khả năng học các khái niệm (concept). Quá trình này tập trung vào việc thiết kế các phương pháp tự động trích xuất thông tin có giá trị từ dữ liệu.
Ba thành phần cốt lõi của Học máy bao gồm:
• Dữ liệu (Data): Được biểu diễn dưới dạng các vectơ số để máy tính có thể xử lý.
• Mô hình (Model): Các hàm số hoặc phân phối xác suất dùng để mô tả quá trình tạo ra dữ liệu hoặc dự đoán kết quả.
• Việc học (Learning): Quá trình tối ưu hóa các tham số của mô hình dựa trên dữ liệu hiện có để thực hiện tốt trên dữ liệu mới.

--------------------------------------------------------------------------------
2. Phân loại Học máy
Dựa trên phương pháp tiếp cận và cách thức học, Học máy được chia thành các nhóm chính:
Theo phương pháp học:
• Phương pháp quy nạp: Máy học dựa trên dữ liệu đã thu thập được trước đó để rút ra quy luật.
• Phương pháp suy diễn: Máy học dựa trên các luật và kiến thức chuyên ngành có sẵn.
Theo nhóm giải thuật:
• Học có giám sát (Supervised Learning): Máy tính học từ các cặp dữ liệu đầu vào và đầu ra (nhãn) có sẵn để dự đoán kết quả cho dữ liệu mới.
• Học không giám sát (Unsupervised Learning): Máy tính tự tìm cấu trúc hoặc cách phân loại dữ liệu từ các mẫu không có nhãn.
• Học nửa giám sát (Semi-supervised Learning): Một dạng lai giữa hai loại trên.
• Học tăng cường (Reinforcement Learning): Máy tính đưa ra hành động và nhận phản hồi (thưởng/phạt) từ môi trường để tự điều chỉnh hành vi.

--------------------------------------------------------------------------------
3. Nền tảng Toán học
Toán học là ngôn ngữ nền tảng để xây dựng và hiểu các thuật toán AI/ML. Có bốn lĩnh vực trọng tâm:
• Đại số tuyến tính: Sử dụng ma trận và vectơ để biểu diễn dữ liệu và các phép biến đổi không gian.
• Giải tích: Cung cấp các công cụ như đạo hàm, gradient, chuỗi Taylor và ma trận Jacobian/Hessian để tối ưu hóa hàm mục tiêu.
• Xác suất và Thống kê: Giúp mô hình hóa sự không chắc chắn và đưa ra quyết định dựa trên dữ liệu (định lý Bayes, phân phối Gaussian).
• Tối ưu hóa: Tìm giá trị nhỏ nhất/lớn nhất của các hàm số thông qua các thuật toán lặp.

--------------------------------------------------------------------------------
4. Các mô hình và Thuật toán tiêu biểu
Cây quyết định (Decision Tree)
Là mô hình dự báo ánh xạ các quan sát về một sự vật tới các kết luận về giá trị mục tiêu.
• Cấu trúc: Gồm các nút (internal nodes) tương ứng với biến, cành (branches) thể hiện kết hợp thuộc tính và lá (leaves) đại diện cho phân loại.
• Thuật toán: ID3 và C4.5 là hai thuật toán quy nạp cây phổ biến, sử dụng các khái niệm như Entropy và Thông tin thu được (Information Gain) để chọn thuộc tính phân loại tốt nhất.
Mạng nơ-ron nhân tạo (ANN)
Mô phỏng cách xử lý thông tin của hệ thần kinh sinh học, gồm nhiều tầng (input, hidden, output) kết nối qua các trọng số.
• Học: Bản chất là quá trình hiệu chỉnh trọng số để tối thiểu hóa hàm lỗi bằng kỹ thuật Lan truyền ngược (Backpropagation).
Các thuật toán tối ưu (Optimizers)
Đóng vai trò cải thiện trọng số và bias theo từng bước:
• Gradient Descent (GD): Di chuyển ngược hướng đạo hàm để tìm cực tiểu.
• Stochastic Gradient Descent (SGD): Cập nhật trọng số trên từng điểm dữ liệu, phù hợp với dữ liệu lớn và học trực tuyến (online learning).
• Adam, Momentum, Adagrad: Các biến thể nâng cao giúp tăng tốc độ hội tụ và vượt qua các cực tiểu địa phương.

--------------------------------------------------------------------------------
5. Kỹ thuật nâng cao: Thuật toán Vượt khe (Cleft-overstep)
Trong một số bài toán phức tạp, mặt sai số có thể xuất hiện dạng lòng khe (ravine) hẹp, khiến các thuật toán Gradient truyền thống bị tắc hoặc hội tụ chậm.
• Nguyên lý: Điều chỉnh độ dài bước học sao cho điểm tìm kiếm luôn bước qua hai phía của lòng khe, tránh bị rơi vào đáy khe quá sớm trước khi đạt lời giải tối ưu.
• Kết hợp: Việc kết hợp Giải thuật di truyền (Genetic Algorithm - GA) để tìm bộ trọng số khởi tạo tốt và Thuật toán vượt khe giúp tăng đáng kể độ chính xác và tốc độ hội tụ cho mạng nơ-ron.

--------------------------------------------------------------------------------
6. Ứng dụng của Học máy
Học máy có ứng dụng rộng khắp trong đời sống và sản xuất:
• Xử lý ngôn ngữ tự nhiên (NLP): Giao tiếp người - máy, dịch thuật tự động.
• Nhận dạng: Tiếng nói, chữ viết tay, khuôn mặt và thị giác máy tính.
• Y tế: Chẩn đoán bệnh tự động, phân tích ảnh X-quang.
• Tài chính: Phát hiện gian lận thẻ tín dụng, phân tích thị trường chứng khoán.
• Giáo dục: Cá nhân hóa lộ trình học tập, hệ thống dạy học thông minh (ITS) giúp chẩn đoán vấn đề của từng người học.



