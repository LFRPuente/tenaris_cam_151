from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cam151Config:
    green_h_lo: int = 35
    green_s_lo: int = 35
    green_v_lo: int = 35
    green_h_hi: int = 95
    green_s_hi: int = 255
    green_v_hi: int = 255
    green_open_size: int = 5
    green_close_size: int = 11
    min_green_area: int = 1500
    bottom_roi_y_frac: float = 0.66
    canny_low: int = 40
    canny_high: int = 120
    hough_threshold: int = 55
    hough_min_line_length: int = 120
    hough_max_line_gap: int = 20
    target_green_x_margin_left: int = 20
    target_green_x_margin_right: int = 220
    target_green_y_margin_top: int = 140
    target_green_y_margin_bottom: int = 140
    post_red_h_lo_1: int = 0
    post_red_h_hi_1: int = 12
    post_red_h_lo_2: int = 160
    post_red_h_hi_2: int = 179
    post_orange_h_lo: int = 8
    post_orange_h_hi: int = 30
    post_s_lo: int = 50
    post_v_lo: int = 50
    post_open_size: int = 3
    post_close_width: int = 3
    post_close_height: int = 7
    post_min_area: int = 40
    post_min_height: int = 15
    post_min_aspect: float = 1.5
