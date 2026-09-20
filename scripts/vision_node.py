#!/usr/bin/env python3
"""
vision_node.py -- HEROES GMRT Vision Node

Menjalankan:
    ros2 run gmrt_vision vision_node.py --ros-args -p source:=0
    ros2 run gmrt_vision vision_node.py --ros-args -p source:=sample_videos/tower_demo.mp4
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from ultralytics import YOLO

from gmrt_vision.msg import TowerStatus


def identifikasi_sblock(crop_bgr):
    """OpenCV klasik (HSV) untuk menentukan ID pola Sky Block (0-6)."""
    if crop_bgr is None or crop_bgr.shape[0] == 0 or crop_bgr.shape[1] == 0:
        return 0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    mask_biru = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    mask_merah1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_merah2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    mask_merah = mask_merah1 | mask_merah2

    total_pixel = crop_bgr.shape[0] * crop_bgr.shape[1]
    px_biru = cv2.countNonZero(mask_biru)
    px_merah = cv2.countNonZero(mask_merah)

    if px_biru < 0.1 * total_pixel and px_merah < 0.1 * total_pixel:
        return 0

    if px_biru > 0.75 * (px_biru + px_merah):
        return 1
    if px_merah > 0.75 * (px_biru + px_merah):
        return 2

    h, w = crop_bgr.shape[:2]
    h2, w2 = h // 2, w // 2

    biru_atas = cv2.countNonZero(mask_biru[:h2, :])
    biru_bawah = cv2.countNonZero(mask_biru[h2:, :])
    biru_kiri = cv2.countNonZero(mask_biru[:, :w2])
    biru_kanan = cv2.countNonZero(mask_biru[:, w2:])

    merah_atas = cv2.countNonZero(mask_merah[:h2, :])
    merah_bawah = cv2.countNonZero(mask_merah[h2:, :])
    merah_kiri = cv2.countNonZero(mask_merah[:, :w2])
    merah_kanan = cv2.countNonZero(mask_merah[:, w2:])

    if biru_atas > merah_atas and merah_bawah > biru_bawah:
        return 3  # biru di atas
    if biru_bawah > merah_bawah and merah_atas > biru_atas:
        return 5  # biru di bawah
    if biru_kanan > merah_kanan and merah_kiri > biru_kiri:
        return 4  # biru di kanan
    if biru_kiri > merah_kiri and merah_kanan > biru_kanan:
        return 6  # biru di kiri

    return 0


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('source', '0')
        self.declare_parameter('model_path', 'yolo_kotak_3d.pt')
        self.declare_parameter('det_conf', 0.6)
        self.declare_parameter('min_box_area_frac', 0.02)
        self.declare_parameter('x_tolerance_frac', 0.15)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('debug_view', False)   # matikan default -- aman utk robot headless

        source_str = self.get_parameter('source').get_parameter_value().string_value
        try:
            self.source = int(source_str)      # angka -> index webcam
        except ValueError:
            self.source = source_str            # bukan angka -> path video

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.det_conf = self.get_parameter('det_conf').get_parameter_value().double_value
        self.min_area_frac = self.get_parameter('min_box_area_frac').get_parameter_value().double_value
        self.x_tol_frac = self.get_parameter('x_tolerance_frac').get_parameter_value().double_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        rate_hz = max(rate_hz, 5.0)   # requirement README: minimal 5Hz
        self.debug_view = self.get_parameter('debug_view').get_parameter_value().bool_value

        self.get_logger().info(f"Memuat model YOLO dari '{model_path}' ...")
        self.model = YOLO(model_path)

        self.get_logger().info(f"Membuka sumber video: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.get_logger().error(f"GAGAL membuka sumber video: {self.source}")

        self.publisher_ = self.create_publisher(TowerStatus, '/vision/tower_status', 10)

        self.timer = self.create_timer(1.0 / rate_hz, self.timer_callback)
        self.get_logger().info(f"vision_node siap, publish rate {rate_hz:.1f} Hz")

    def _kelompokkan_satu_tower(self, daftar_kotak, lebar_layar):
        """Buang deteksi yang X-nya jauh dari tower utama (background/ngasal)."""
        if len(daftar_kotak) <= 1:
            return daftar_kotak
        cxs = np.array([k[1] for k in daftar_kotak], dtype=np.float32)
        median_cx = np.median(cxs)
        toleransi = self.x_tol_frac * lebar_layar
        return [k for k in daftar_kotak if abs(k[1] - median_cx) <= toleransi]

    def timer_callback(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            # video file habis -> ulang dari awal (webcam: ret=False jarang terjadi)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        tinggi_layar, lebar_layar = frame.shape[:2]

        hasil = self.model.track(frame, conf=self.det_conf, persist=True, verbose=False)

        daftar_kotak = []
        for r in hasil:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                box = r.boxes.xyxy[i].cpu().numpy().astype(int)
                luas_frac = ((box[2] - box[0]) * (box[3] - box[1])) / (lebar_layar * tinggi_layar)
                if luas_frac < self.min_area_frac:
                    continue

                nama = self.model.names[int(r.boxes.cls[i].item())]
                kode_warna = 1 if nama == "kotak_biru" else 2
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                daftar_kotak.append((cy, cx, box, kode_warna))

        daftar_kotak = self._kelompokkan_satu_tower(daftar_kotak, lebar_layar)

        eblock1 = eblock2 = sblock = 0

        if daftar_kotak:
            daftar_kotak.sort(key=lambda k: k[0], reverse=True)   # bawah dulu

            if len(daftar_kotak) >= 1:
                eblock1 = daftar_kotak[0][3]
            if len(daftar_kotak) >= 2:
                eblock2 = daftar_kotak[1][3]
            if len(daftar_kotak) >= 3:
                bx = daftar_kotak[2][2]
                crop = frame[bx[1]:bx[3], bx[0]:bx[2]]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    sblock = identifikasi_sblock(crop)

        msg = TowerStatus()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.eblock1 = eblock1
        msg.eblock2 = eblock2
        msg.sblock = sblock
        self.publisher_.publish(msg)

        # Visualisasi debug: Menggambar kotak manual OpenCV yang rapi dan panel HUD
        if self.debug_view:
            vis = frame.copy()
            pusat_x, pusat_y = lebar_layar // 2, tinggi_layar // 2
            cv2.drawMarker(vis, (pusat_x, pusat_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            for idx, (cy, cx, box, warna_id) in enumerate(daftar_kotak):
                warna_garis = (255, 0, 0) if warna_id == 1 else (0, 0, 255) # Biru / Merah
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), warna_garis, 2)
                cv2.circle(vis, (cx, cy), 5, (255, 255, 255), -1)
                cv2.line(vis, (pusat_x, pusat_y), (cx, cy), (200, 200, 200), 1)

                teks_posisi = f"Level {idx} | ID:{warna_id}"
                cv2.putText(vis, teks_posisi, (box[0], max(20, box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_garis, 2)

            # Panel informasi status tower di pojok kiri atas video
            panel = np.zeros((120, 320, 3), dtype=np.uint8)
            cv2.putText(panel, f"eblock1 : {eblock1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, f"eblock2 : {eblock2}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, f"sblock  : {sblock}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            vis[0:120, 0:320] = panel

            cv2.imshow("GMRT Vision Node (debug)", cv2.resize(vis, (640, 480)))
            cv2.waitKey(1)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        if self.debug_view:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
