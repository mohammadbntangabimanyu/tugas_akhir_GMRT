# GMRT Vision Package (`gmrt_vision`)
### HEROES Gadjah Mada Robotic Team (GMRT)

Repositori ini berisi *ROS 2 package* untuk sistem *Computer Vision* robot GMRT. Node ini bertugas untuk membaca *frame* dari kamera atau video sampel, mendeteksi tumpukan *Earth Block* dan *Sky Block* pada sebuah *Building Spot*, lalu mempublikasikan statusnya ke jaringan ROS 2.

---

## 📋 Fitur Utama
1. **ROS 2 Node Mandiri (`vision_node`)**: Menggabungkan pemrosesan gambar dan publikasi data dalam satu *node* efisien yang berjalan minimal pada 5 Hz.
2. **Dinamis Parameter (`source`)**: Mendukung input dari indeks *webcam* fisik (misal: `0`) maupun file video simulasi tanpa perlu mengubah baris kode.
3. **Deteksi Berbasis YOLO & HSV**: Menggunakan model YOLO untuk mendeteksi kotak-kotak kubus 3D serta pengolahan citra klasik OpenCV (HSV) untuk mengklasifikasikan pola orientasi *Sky Block* (`sblock`) ke dalam 7 kategori (ID 0–6).
4. **Visualisasi Debug Bersih**: Dilengkapi mode tampilan opsional (`debug_view`) dengan kotak pembatas (*bounding box*) manual OpenCV yang akurat dan bebas dari gangguan garis *pose* acak.

---

## 📂Struktur Package
```text
gmrt_vision/
├── gmrt_vision/          # Python module folder
├── msg/
│   └── TowerStatus.msg   # Definisi custom message ROS 2
├── scripts/
│   └── vision_node.py    # Skrip utama ROS 2 Node Vision
├── sample_videos/
│   └── tower_demo.mp4    # Video sampel pengujian
├── CMakeLists.txt
└── package.xml
