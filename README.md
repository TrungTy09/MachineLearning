Bên trên chứa các file cần thiết để có thể train 1 model nhận diện mệnh giá tiền với độ chính xác cao bằng VGG16. 
Để đảm bảo code không xảy ra lỗi gì thì bạn nên cài đặt đủ các thư viện cần thiết
// Trong file take_data.py 
bạn sẽ cần phải tạo ra 1 thư mục data/ để chứa các folder con là các mệnh giá tiền 
// Lưu ý để việc sắp xếp các nhãn theo đúng trật tự thì số lượng ký tự trong nhãn(label) phải bằng nhau 
ví dụ: 
label:"000000"
label:"001000"
label:"002000"
// Tiếp theo đến file train nếu là lần đầu bạn chạy file này thì bạn sẽ phải chạy hàm save_data() 
// Còn nếu từ lần thứ 2 trở đi bạn chỉ việc gọi hàm load_data() để lấy thông tin từ file pix.data
sau khi quá trình train hoàn tất bạn sẽ có 1 file weight.h5 bạn sẽ dùng file này để test lại với file test.py
