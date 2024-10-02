
# Các tính năng của LightGBM

Đây là cái nhìn tổng quan khái niệm về cách LightGBM hoạt động. Chúng tôi giả định rằng bạn đã quen thuộc với các thuật toán tăng cường cây quyết định để tập trung vào các khía cạnh của LightGBM có thể khác so với các gói tăng cường khác. Để biết chi tiết về các thuật toán, vui lòng tham khảo các trích dẫn hoặc mã nguồn.

## Tối ưu hóa về tốc độ và sử dụng bộ nhớ

Nhiều công cụ tăng cường sử dụng các thuật toán dựa trên tiền sắp xếp (pre-sort) \[2, 3\], chẳng hạn như thuật toán mặc định trong XGBoost, cho việc học cây quyết định. Đây là một giải pháp đơn giản, nhưng không dễ tối ưu hóa.

LightGBM sử dụng các thuật toán dựa trên histogram \[4, 5, 6\], trong đó các giá trị đặc trưng liên tục được phân thành các thùng rời rạc. Điều này giúp tăng tốc độ huấn luyện và giảm sử dụng bộ nhớ. Các lợi thế của thuật toán dựa trên histogram bao gồm:

- **Giảm chi phí tính toán độ lợi cho mỗi lần chia**: Thuật toán dựa trên tiền sắp xếp có độ phức tạp thời gian là O(#data). Việc tính toán histogram có độ phức tạp là O(#data), nhưng điều này chỉ bao gồm các phép tính tổng nhanh. Sau khi histogram được xây dựng, thuật toán dựa trên histogram có độ phức tạp thời gian là O(#bins), và #bins nhỏ hơn nhiều so với #data.

- **Sử dụng phép trừ histogram để tăng tốc**: Để có được các histogram của một lá trong cây nhị phân, sử dụng phép trừ histogram của lá cha và lá lân cận của nó. Điều này chỉ cần xây dựng histogram cho một lá, sau đó có thể lấy histogram của lá lân cận bằng cách trừ với chi phí nhỏ (O(#bins)).

- **Giảm sử dụng bộ nhớ**: Thay thế các giá trị liên tục bằng các thùng rời rạc. Nếu #bins nhỏ, có thể sử dụng kiểu dữ liệu nhỏ, chẳng hạn như uint8_t, để lưu trữ dữ liệu huấn luyện. Không cần lưu trữ thêm thông tin để sắp xếp trước các giá trị đặc trưng.

- **Giảm chi phí giao tiếp cho việc học phân tán**

## Tối ưu hóa độ chính xác

### Tăng trưởng cây theo lá (Leaf-wise)

Hầu hết các thuật toán học cây quyết định đều tăng trưởng cây theo mức (depth-wise). LightGBM phát triển cây theo lá (leaf-wise). Nó sẽ chọn lá có thay đổi độ mất mát (delta loss) lớn nhất để phát triển. Khi giữ cố định #leaf, các thuật toán theo lá có xu hướng đạt được độ mất mát thấp hơn so với thuật toán theo mức.

Việc tăng trưởng theo lá có thể gây ra overfitting khi #data nhỏ, do đó LightGBM bao gồm tham số max_depth để giới hạn độ sâu của cây. Tuy nhiên, cây vẫn phát triển theo lá ngay cả khi max_depth được chỉ định.

## Phân chia tối ưu cho các đặc trưng phân loại

Thông thường, các đặc trưng phân loại được biểu diễn bằng one-hot encoding, nhưng phương pháp này không tối ưu cho việc học cây. Đặc biệt đối với các đặc trưng phân loại có số lượng loại lớn, một cây xây dựng trên các đặc trưng one-hot có xu hướng không cân bằng và cần phát triển rất sâu để đạt được độ chính xác tốt.

Thay vì sử dụng one-hot encoding, giải pháp tối ưu là chia một đặc trưng phân loại bằng cách phân chia các loại của nó thành hai tập con. Nếu đặc trưng có k loại, có 2^(k-1) - 1 phân chia có thể. Tuy nhiên, có một giải pháp hiệu quả cho các cây hồi quy \[8\]. Nó cần khoảng O(k * log(k)) để tìm phân chia tối ưu.

## Tối ưu hóa trong giao tiếp mạng

LightGBM chỉ cần sử dụng một số thuật toán giao tiếp tập thể, như "All reduce", "All gather" và "Reduce scatter", trong học phân tán. LightGBM triển khai các thuật toán tiên tiến nhất \[9\].

## Tối ưu hóa trong học phân tán

LightGBM cung cấp các thuật toán học phân tán sau:

### Feature Parallel trong LightGBM

Vì feature parallel không thể tăng tốc khi #data lớn, LightGBM thực hiện một thay đổi nhỏ: thay vì phân vùng dữ liệu theo chiều dọc, mỗi worker giữ toàn bộ dữ liệu. Điều này giúp không cần giao tiếp kết quả chia tách dữ liệu vì mỗi worker đều biết cách chia tách. Tuy nhiên, thuật toán này vẫn có chi phí tính toán cho "split" khi #data lớn.

### Data Parallel trong LightGBM

Data parallel nhằm mục đích phân vùng theo chiều ngang dữ liệu. Để giảm chi phí giao tiếp, LightGBM sử dụng "Reduce Scatter" để hợp nhất các histogram của các đặc trưng khác nhau cho các worker khác nhau, sau đó các worker tìm điểm chia cục bộ tốt nhất trên các histogram hợp nhất và đồng bộ hóa điểm chia toàn cục tốt nhất.

### Voting Parallel

Voting parallel giảm chi phí giao tiếp trong Data Parallel thành chi phí hằng số. Nó sử dụng bỏ phiếu hai giai đoạn để giảm chi phí giao tiếp của các histogram đặc trưng \[10\].

## Hỗ trợ GPU

Cảm ơn @huanzhang12 đã đóng góp tính năng này.

## Ứng dụng và các phép đo

LightGBM hỗ trợ các ứng dụng sau:

- Hồi quy
- Phân loại nhị phân
- Phân loại đa lớp
- LambdaRank
- Cross-entropy

LightGBM hỗ trợ các phép đo sau:

- L1, L2 loss
- Log loss
- Classification error rate
- AUC
- NDCG
- MAP
- Multi-class log loss
- Multi-class error rate

## Các tính năng khác

- Giới hạn độ sâu tối đa của cây khi phát triển theo lá
- Hỗ trợ DART
- Regularization L1/L2
- Bagging
- Sub-sample đặc trưng
- Train liên tục với mô hình GBDT đầu vào
- Early stopping
- Nhiều phép đo xác thực

## Tài liệu tham khảo

\[1\] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NIPS 2017.

\[2\] Mehta, M., Agrawal, R., and Rissanen, J. "SLIQ: A fast scalable classifier for data mining." EDBT 1996.

\[3\] Shafer, J., Agrawal, R., and Mehta, M. "SPRINT: A scalable parallel classifier for data mining." VLDB 1996.

\[4\] Ranka, S., and Singh, V. "CLOUDS: A decision tree classifier for large datasets." KDD 1998.

\[5\] Machado, F. P. "Communication and memory efficient parallel decision tree construction." 2003.

\[6\] Li, P., Wu, Q., and Burges, C. J. "Mcrank: Learning to rank using multiple classification and gradient boosting." NIPS 2007.

\[7\] Shi, H. "Best-first decision tree learning." University of Waikato 2007.

\[8\] Fisher, W. D. "On Grouping for Maximum Homogeneity." Journal of the American Statistical Association 1958.

\[9\] Thakur, R., Rabenseifner, R., and Gropp, W. "Optimization of collective communication operations in MPICH." IJHPCA 2005.

\[10\] Meng, Q., Ke, G., Wang, T., Chen, W., Ye, Q., Ma, Z., and Liu, T. "A Communication-Efficient Parallel Algorithm for Decision Tree." NIPS 2016.
