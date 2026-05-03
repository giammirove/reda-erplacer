use crate::placement::PlacementSolution;
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
use reda_db::{Numeric, DB};

// 5x7 bitmap font for digits 0-9, each digit is 5 columns of 7 bits (MSB = top row)
const DIGIT_BITMAPS: [[u8; 5]; 10] = [
    [0x7C, 0x8A, 0x92, 0xA2, 0x7C], // 0
    [0x00, 0x42, 0xFE, 0x02, 0x00], // 1
    [0x46, 0x8A, 0x92, 0x92, 0x62], // 2
    [0x44, 0x82, 0x92, 0xB2, 0xCC], // 3
    [0x18, 0x28, 0x48, 0xFE, 0x08], // 4
    [0xE4, 0xA2, 0xA2, 0xA2, 0x9C], // 5
    [0x3C, 0x52, 0x92, 0x92, 0x0C], // 6
    [0x80, 0x8E, 0x90, 0xA0, 0xC0], // 7
    [0x6C, 0x92, 0x92, 0x92, 0x6C], // 8
    [0x60, 0x92, 0x92, 0x94, 0x78], // 9
];

fn draw_number(img: &mut RgbImage, n: usize, cx: i32, cy: i32, scale: u32, color: Rgb<u8>) {
    let s = n.to_string();
    let digits: Vec<usize> = s.chars().map(|c| c as usize - '0' as usize).collect();
    let s = scale as i32;
    let total_w = (digits.len() as i32) * (5 * s + s); // 5px*scale + gap
    let mut dx = cx - total_w / 2;
    for d in digits {
        let bitmap = &DIGIT_BITMAPS[d];
        for col in 0..5usize {
            for row in 0..7usize {
                if bitmap[col] & (1 << (7 - row)) != 0 {
                    for sy in 0..s {
                        for sx in 0..s {
                            let px = dx + col as i32 * s + sx;
                            let py = cy - 3 * s + row as i32 * s + sy;
                            if px >= 0
                                && py >= 0
                                && px < img.width() as i32
                                && py < img.height() as i32
                            {
                                img.put_pixel(px as u32, py as u32, color);
                            }
                        }
                    }
                }
            }
        }
        dx += 6 * s;
    }
}

pub(crate) fn draw_placement<T>(
    db: &DB<T>,
    solution: &PlacementSolution<T>,
    macro_colors: &[[u8; 3]],
    iteration: usize,
) where
    T: Numeric + std::fmt::Debug,
{
    let diearea = &db.diearea;
    let w = diearea.full_width();
    let h = diearea.full_height();
    let k = T::from(0.0006).unwrap();
    // compute scale = k * sqrt(width * height)
    let pl_scale = k * (w * h).sqrt();

    let width: u32 = (w.ceil() / pl_scale).to_u32().unwrap();
    let height: u32 = (h.ceil() / pl_scale).to_u32().unwrap();
    let iwidth: i32 = (w.ceil() / pl_scale).to_i32().unwrap();
    let iheight: i32 = (h.ceil() / pl_scale).to_i32().unwrap();
    let num_movable = db.num_movable;

    let mut img = RgbImage::new(width, height);

    // Optional: fill with white
    for pixel in img.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    let init_instances = &db.instances;

    // Compute macro area threshold (10x median movable area)
    let macro_threshold = if num_movable > 0 {
        let mut sorted: Vec<T> = init_instances.areas[..num_movable].to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2] * T::from(10.0).unwrap()
    } else {
        T::from(f32::MAX).unwrap()
    };

    let instances = &solution.instances;
    let mut macro_idx = 0usize;
    for ((((i, x), y), w), h) in instances
        .x
        .iter()
        .enumerate()
        .zip(&instances.y)
        .zip(&init_instances.sizes.w)
        .zip(&init_instances.sizes.h)
    {
        let w = (*w / pl_scale).ceil().to_u32().unwrap().max(0);
        let h = (*h / pl_scale).ceil().to_u32().unwrap().max(0);

        let x = (*x / pl_scale)
            .ceil()
            .to_i32()
            .unwrap()
            .clamp(0, iwidth - 1);
        let y = (height as i32 - (*y / pl_scale).ceil().to_i32().unwrap() - h as i32)
            .clamp(0, iheight - 1);

        // use x, y, w, h
        let is_macro = i >= num_movable || init_instances.areas[i] > macro_threshold;
        let color = if is_macro {
            if let Some(&[r, g, b]) = macro_colors.get(macro_idx) {
                Rgb([r, g, b])
            } else {
                Rgb([255, 0, 0])
            }
        } else {
            Rgb([0, 0, 255])
        };

        // TODO: dont skip PIN
        if w == 0 || h == 0 {
            continue;
        }

        assert!(x >= 0 && x < iwidth, "0 < x {:07} < {:07}", x, iwidth);
        assert!(y >= 0 && y < iheight, "0 < y {:07} < {:07}", y, iheight);
        let rect = Rect::at(x, y).of_size(w, h);
        draw_filled_rect_mut(&mut img, rect, color);
        if is_macro {
            let cx = x + w as i32 / 2;
            let cy = y + h as i32 / 2;
            let scale = h.min(w).isqrt(); // ~1 digits tall relative to rect

            draw_number(&mut img, macro_idx, cx, cy, scale, Rgb([0, 0, 0]));
            macro_idx += 1;
        }
    }

    // Save to file
    img.save(format!("images/solution_{:07}.png", iteration))
        .expect("Failed to save image");

    log::debug!("Screenshot taken");
}
