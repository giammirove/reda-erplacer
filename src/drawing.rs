use crate::placement::PlacementSolution;
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
use reda_db::{Numeric, DB};

pub(crate) fn draw_placement<T>(db: &DB<T>, solution: &PlacementSolution<T>, iteration: usize)
where
    T: Numeric + std::fmt::Debug,
{
    let diearea = &db.diearea;
    let width: u32 = diearea.width().ceil().to_u32().unwrap();
    let height: u32 = diearea.height().ceil().to_u32().unwrap();
    let num_movable = db.num_movable;

    let mut img = RgbImage::new(width, height);

    // Optional: fill with white
    for pixel in img.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    let init_instances = &db.instances;
    let instances = &solution.instances;
    for ((((i, x), y), w), h) in instances
        .x
        .iter()
        .enumerate()
        .zip(&instances.y)
        .zip(&init_instances.sizes.w)
        .zip(&init_instances.sizes.h)
    {
        let w = (*w).ceil().to_u32().unwrap();
        let h = (*h).ceil().to_u32().unwrap();

        let x = (*x).ceil().to_i32().unwrap();
        let y = height as i32 - (*y).ceil().to_i32().unwrap() - h as i32;

        // use x, y, w, h
        let color = if i >= num_movable {
            Rgb([255, 0, 0])
        } else {
            Rgb([0, 0, 255])
        };

        // TODO: dont skip PIN
        if w == 0 || h == 0 {
            continue;
        }

        let rect = Rect::at(x, y).of_size(w, h);
        draw_filled_rect_mut(&mut img, rect, color);
    }

    // Save to file
    img.save(format!("images/solution_{:07}.png", iteration))
        .expect("Failed to save image");

    log::debug!("Screenshot taken");
}
