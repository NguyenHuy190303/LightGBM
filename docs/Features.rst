Tính năng
=========

Đây là cái nhìn tổng quan khái niệm về cách LightGBM hoạt động `[1] <#references>`. Chúng tôi giả định rằng bạn đã quen thuộc với các thuật toán tăng cường cây quyết định để tập trung vào các khía cạnh của LightGBM có thể khác biệt so với các gói tăng cường khác. Để biết chi tiết về các thuật toán, vui lòng tham khảo các tài liệu trích dẫn hoặc mã nguồn.

Tối ưu hóa về tốc độ và sử dụng bộ nhớ
--------------------------------------

Nhiều công cụ tăng cường sử dụng các thuật toán dựa trên sắp xếp trước `[2, 3] <#references>` (ví dụ, thuật toán mặc định trong xgboost) để học cây quyết định. Đây là một giải pháp đơn giản, nhưng không dễ tối ưu hóa.

LightGBM sử dụng các thuật toán dựa trên biểu đồ tần suất `[4, 5, 6] <#references>`, trong đó chia giá trị của các thuộc tính liên tục thành các nhóm rời rạc. Điều này giúp tăng tốc độ huấn luyện và giảm sử dụng bộ nhớ. Các lợi thế của các thuật toán dựa trên biểu đồ tần suất bao gồm:

-  **Giảm chi phí tính toán độ lợi cho mỗi lần chia**

   -  Các thuật toán dựa trên sắp xếp trước có độ phức tạp thời gian là ``O(#data)``

   -  Tính toán biểu đồ tần suất có độ phức tạp thời gian là ``O(#data)``, nhưng chỉ cần một phép cộng nhanh chóng. Khi biểu đồ đã được xây dựng, thuật toán dựa trên biểu đồ tần suất có độ phức tạp thời gian là ``O(#bins)``, và ``#bins`` nhỏ hơn nhiều so với ``#data``.

-  **Sử dụng phép trừ biểu đồ tần suất để tăng tốc độ hơn nữa**

   -  Để lấy biểu đồ tần suất của một lá trong cây nhị phân, sử dụng phép trừ biểu đồ của cha và lá kề của nó

   -  Vì vậy, chỉ cần xây dựng biểu đồ tần suất cho một lá (với ``#data`` nhỏ hơn lá kề). Sau đó có thể lấy biểu đồ tần suất của lá kề bằng phép trừ biểu đồ với chi phí nhỏ (``O(#bins)``)

-  **Giảm sử dụng bộ nhớ**

   -  Thay thế các giá trị liên tục bằng các nhóm rời rạc. Nếu ``#bins`` nhỏ, có thể sử dụng kiểu dữ liệu nhỏ, ví dụ như uint8_t, để lưu trữ dữ liệu huấn luyện

   -  Không cần lưu trữ thông tin bổ sung cho việc sắp xếp trước các giá trị thuộc tính

-  **Giảm chi phí truyền thông cho học phân tán**

Tối ưu hóa tính thưa
-------------------

-  Chỉ cần ``O(2 * #non_zero_data)`` để xây dựng biểu đồ tần suất cho các thuộc tính thưa

Tối ưu hóa độ chính xác
------------------------

Phát triển cây theo lá (Best-first)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hầu hết các thuật toán học cây quyết định phát triển cây theo từng mức (depth-wise), như trong hình sau:

.. image:: ./_static/images/level-wise.png
   :align: center
   :alt: Sơ đồ minh họa sự phát triển của cây theo từng mức, trong đó nút tốt nhất có thể được chia nhỏ xuống một mức. Chiến lược này tạo ra một cây đối xứng, nơi mỗi nút trong một mức có các nút con, dẫn đến một lớp độ sâu bổ sung.

LightGBM phát triển cây theo từng lá (best-first) `[7] <#references>`. Nó sẽ chọn lá có delta mất mát lớn nhất để phát triển.
Khi giữ cố định ``#leaf``, các thuật toán phát triển theo lá thường đạt được mất mát thấp hơn so với các thuật toán phát triển theo mức.

Phát triển theo lá có thể gây ra hiện tượng over-fitting khi ``#data`` nhỏ, vì vậy LightGBM bao gồm tham số ``max_depth`` để giới hạn độ sâu của cây. Tuy nhiên, cây vẫn sẽ phát triển theo lá ngay cả khi đã chỉ định ``max_depth``.

.. image:: ./_static/images/leaf-wise.png
   :align: center
   :alt: Sơ đồ minh họa sự phát triển của cây theo từng lá, trong đó chỉ có nút có thay đổi mất mát cao nhất được chia nhỏ mà không cần quan tâm đến các nút còn lại trong cùng mức. Điều này tạo ra một cây không đối xứng, nơi việc chia nhỏ tiếp theo chỉ diễn ra ở một phía của cây.

Chia tối ưu cho các thuộc tính phân loại
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thông thường, các thuộc tính phân loại được biểu diễn bằng mã hóa one-hot, nhưng phương pháp này không tối ưu cho các bộ học cây. Đặc biệt là với các thuộc tính phân loại có số hạng lớn, một cây dựa trên các thuộc tính one-hot có xu hướng không cân bằng và cần phát triển rất sâu để đạt được độ chính xác tốt.

Thay vì sử dụng mã hóa one-hot, giải pháp tối ưu là chia một thuộc tính phân loại bằng cách chia các danh mục của nó thành 2 tập hợp con. Nếu thuộc tính có ``k`` danh mục, có ``2^(k-1) - 1`` cách chia có thể.
Tuy nhiên, có một giải pháp hiệu quả cho các cây hồi quy `[8] <#references>`. Nó cần khoảng ``O(k * log(k))`` để tìm được cách chia tối ưu.

Ý tưởng cơ bản là sắp xếp các danh mục theo mục tiêu huấn luyện tại mỗi lần tách.
Cụ thể hơn, LightGBM sắp xếp histogram (đối với một đặc trưng phân loại) theo các giá trị tích lũy của nó (``sum_gradient / sum_hessian``) và sau đó tìm điểm tách tốt nhất trên histogram đã sắp xếp.

Tối ưu hóa trong truyền thông mạng
-----------------------------------

Chỉ cần sử dụng một số thuật toán truyền thông tập thể, như "All reduce", "All gather" và "Reduce scatter", trong quá trình học phân tán của LightGBM.
LightGBM triển khai các thuật toán tiên tiến nhất\ `[9] <#references>`__.
Các thuật toán truyền thông tập thể này có thể mang lại hiệu suất tốt hơn so với truyền thông điểm-điểm.

.. _Tối ưu hóa trong học song song:

Tối ưu hóa trong học phân tán
------------------------------------

LightGBM cung cấp các thuật toán học phân tán sau đây.

Feature Parallel
~~~~~~~~~~~~~~~~

Thuật toán truyền thống
^^^^^^^^^^^^^^^^^^^^^

Feature parallel nhằm mục tiêu song song hóa quá trình "Tìm điểm tách tốt nhất" trong cây quyết định. Quy trình của feature parallel truyền thống là:

1. Phân chia dữ liệu theo chiều dọc (các máy khác nhau có tập hợp đặc trưng khác nhau).

2. Các worker tìm điểm tách tốt nhất {feature, threshold} trên tập hợp đặc trưng cục bộ.

3. Trao đổi các điểm tách cục bộ với nhau và tìm điểm tốt nhất.

4. Worker với điểm tách tốt nhất thực hiện tách, sau đó gửi kết quả tách của dữ liệu đến các worker khác.

5. Các worker khác tách dữ liệu theo dữ liệu nhận được.

Những hạn chế của feature parallel truyền thống:

-  Có chi phí tính toán, vì không thể tăng tốc quá trình "tách", có độ phức tạp là ``O(#data)``.
   Do đó, feature parallel không thể tăng tốc tốt khi ``#data`` lớn.

-  Cần trao đổi kết quả tách, chi phí khoảng ``O(#data / 8)`` (một bit cho một dữ liệu).

Feature Parallel trong LightGBM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vì feature parallel không thể tăng tốc tốt khi ``#data`` lớn, chúng tôi thực hiện một chút thay đổi: thay vì phân chia dữ liệu theo chiều dọc, mỗi worker giữ toàn bộ dữ liệu.
Do đó, LightGBM không cần trao đổi kết quả tách của dữ liệu vì mỗi worker đều biết cách tách dữ liệu.
Và ``#data`` không lớn hơn, vì vậy việc giữ toàn bộ dữ liệu trong mỗi máy là hợp lý.

Quy trình của feature parallel trong LightGBM:

1. Các worker tìm điểm tách tốt nhất {feature, threshold} trên tập hợp đặc trưng cục bộ.

2. Trao đổi các điểm tách cục bộ với nhau và tìm điểm tốt nhất.

3. Thực hiện điểm tách tốt nhất.

Tuy nhiên, thuật toán feature parallel này vẫn gặp phải chi phí tính toán cho quá trình "tách" khi ``#data`` lớn.
Do đó sẽ tốt hơn nếu sử dụng data parallel khi ``#data`` lớn.

Data Parallel
~~~~~~~~~~~~~

Thuật toán truyền thống
^^^^^^^^^^^^^^^^^^^^^

Data parallel nhằm mục tiêu song song hóa toàn bộ quá trình học quyết định. Quy trình của data parallel là:

1. Phân chia dữ liệu theo chiều ngang.

2. Các worker sử dụng dữ liệu cục bộ để xây dựng các histogram cục bộ.

3. Hợp nhất các histogram toàn cục từ tất cả các histogram cục bộ.

4. Tìm điểm tách tốt nhất từ các histogram toàn cục đã hợp nhất, sau đó thực hiện tách.

Những hạn chế của data parallel truyền thống:

-  Chi phí truyền thông cao.
   Nếu sử dụng thuật toán truyền thông điểm-điểm, chi phí truyền thông cho một máy khoảng ``O(#machine * #feature * #bin)``.
   Nếu sử dụng thuật toán truyền thông tập thể (ví dụ như "All Reduce"), chi phí truyền thông khoảng ``O(2 * #feature * #bin)`` (kiểm tra chi phí của "All Reduce" trong chương 4.5 tại `[9] <#references>`__).

Data Parallel trong LightGBM
^^^^^^^^^^^^^^^^^^^^^^^^^

Chúng tôi giảm chi phí truyền thông của data parallel trong LightGBM:

1. Thay vì "Hợp nhất các histogram toàn cục từ tất cả các histogram cục bộ", LightGBM sử dụng "Reduce Scatter" để hợp nhất các histogram của các đặc trưng khác nhau (không trùng lặp) cho các worker khác nhau.
   Sau đó các worker tìm điểm tách tốt nhất cục bộ trên các histogram đã hợp nhất và đồng bộ hóa điểm tách tốt nhất toàn cục.

2. Như đã đề cập trước đó, LightGBM sử dụng phép trừ histogram để tăng tốc quá trình huấn luyện.
   Dựa trên điều này, chúng ta có thể chỉ cần truyền thông các histogram cho một leaf, và lấy histogram của lá kế cận bằng phép trừ.

Tổng quan, data parallel trong LightGBM có độ phức tạp tính toán là ``O(0.5 * #feature * #bin)``.

Voting Parallel
~~~~~~~~~~~~~~~

Voting parallel tiếp tục giảm chi phí truyền thông trong `Data Parallel <#data-parallel>`__ xuống chi phí cố định.
Nó sử dụng quy trình bỏ phiếu hai giai đoạn để giảm chi phí truyền thông của các histogram đặc trưng\ `[10] <#references>`__.

Hỗ trợ GPU
-----------

Cảm ơn `@huanzhang12 <https://github.com/huanzhang12>`__ đã đóng góp tính năng này. Vui lòng đọc `[11] <#references>`__ để biết thêm chi tiết.

- `Cài đặt GPU <./Installation-Guide.rst#build-gpu-version>`__

- `Hướng dẫn GPU <./GPU-Tutorial.rst>`__

Ứng dụng và Các Metric
------------------------

LightGBM hỗ trợ các ứng dụng sau:

-  hồi quy, hàm mục tiêu là L2 loss

-  phân loại nhị phân, hàm mục tiêu là logloss

-  phân loại đa lớp

-  cross-entropy, hàm mục tiêu là logloss và hỗ trợ huấn luyện trên các nhãn không phải nhị phân

-  LambdaRank, hàm mục tiêu là LambdaRank với NDCG

LightGBM hỗ trợ các metric sau:

-  L1 loss

-  L2 loss

-  Log loss

-  Tỉ lệ lỗi phân loại

-  AUC

-  NDCG

-  MAP

-  Multi-class log loss

-  Tỉ lệ lỗi đa lớp

-  AUC-mu ``(mới trong v3.0.0)``

-  Average precision ``(mới trong v3.1.0)``

-  Fair

-  Huber

-  Poisson

-  Quantile

-  MAPE

-  Kullback-Leibler

-  Gamma

-  Tweedie

Để biết thêm chi tiết, vui lòng tham khảo `Parameters <./Parameters.rst#metric-parameters>`__.

Các Tính Năng Khác
--------------

-  Giới hạn ``max_depth`` của cây trong khi phát triển cây theo chiều lá

-  `DART <https://arxiv.org/abs/1505.01866>`__

-  L1/L2 regularization

-  Bagging

-  Cắt giảm cột (đặc trưng)

-  Tiếp tục huấn luyện với mô hình GBDT đầu vào

-  Tiếp tục huấn luyện với tệp điểm đầu vào

-  Huấn luyện có trọng số

-  Đầu ra metric đánh giá trong quá trình huấn luyện

-  Hỗ trợ nhiều dữ liệu đánh giá

-  Hỗ trợ nhiều metric

-  Dừng sớm (cả trong huấn luyện và dự đoán)

-  Dự đoán cho chỉ số lá

Để biết thêm chi tiết, vui lòng tham khảo `Parameters <./Parameters.rst>`__.


References
----------

[1] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "`LightGBM\: A Highly Efficient Gradient Boosting Decision Tree`_." Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

[2] Mehta, Manish, Rakesh Agrawal, and Jorma Rissanen. "SLIQ: A fast scalable classifier for data mining." International Conference on Extending Database Technology. Springer Berlin Heidelberg, 1996.

[3] Shafer, John, Rakesh Agrawal, and Manish Mehta. "SPRINT: A scalable parallel classifier for data mining." Proc. 1996 Int. Conf. Very Large Data Bases. 1996.

[4] Ranka, Sanjay, and V. Singh. "CLOUDS: A decision tree classifier for large datasets." Proceedings of the 4th Knowledge Discovery and Data Mining Conference. 1998.

[5] Machado, F. P. "Communication and memory efficient parallel decision tree construction." (2003).

[6] Li, Ping, Qiang Wu, and Christopher J. Burges. "Mcrank: Learning to rank using multiple classification and gradient boosting." Advances in Neural Information Processing Systems 20 (NIPS 2007).

[7] Shi, Haijian. "Best-first decision tree learning." Diss. The University of Waikato, 2007.

[8] Walter D. Fisher. "`On Grouping for Maximum Homogeneity`_." Journal of the American Statistical Association. Vol. 53, No. 284 (Dec., 1958), pp. 789-798.

[9] Thakur, Rajeev, Rolf Rabenseifner, and William Gropp. "`Optimization of collective communication operations in MPICH`_." International Journal of High Performance Computing Applications 19.1 (2005), pp. 49-66.

[10] Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. "`A Communication-Efficient Parallel Algorithm for Decision Tree`_." Advances in Neural Information Processing Systems 29 (NIPS 2016), pp. 1279-1287.

[11] Huan Zhang, Si Si and Cho-Jui Hsieh. "`GPU Acceleration for Large-scale Tree Boosting`_." SysML Conference, 2018.

.. _LightGBM\: A Highly Efficient Gradient Boosting Decision Tree: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf

.. _On Grouping for Maximum Homogeneity: https://www.tandfonline.com/doi/abs/10.1080/01621459.1958.10501479

.. _Optimization of collective communication operations in MPICH: https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf

.. _A Communication-Efficient Parallel Algorithm for Decision Tree: http://papers.nips.cc/paper/6381-a-communication-efficient-parallel-algorithm-for-decision-tree

.. _GPU Acceleration for Large-scale Tree Boosting: https://arxiv.org/abs/1706.08359
