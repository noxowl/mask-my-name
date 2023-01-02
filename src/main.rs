use std::env;
use std::path::PathBuf;
use std::ffi::{CStr, CString, c_int};
use anyhow::{anyhow, Result, Error};
use opencv::core::{in_range, Point, Rect, Scalar, Size};
use opencv::prelude::*;
use opencv::imgcodecs::{imread, IMREAD_UNCHANGED, imwrite};
use opencv::imgproc::{COLOR_BGR2GRAY, COLOR_BGR2HSV, COLOR_HSV2BGR, cvt_color, dilate, MORPH_RECT, get_structuring_element, morphology_default_border_value, find_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, bounding_rect, rectangle, CHAIN_APPROX_NONE};
use opencv::types::VectorOfVectorOfPoint;
use tesseract_plumbing::{TessBaseApi, Text};


fn load_image(image_path: PathBuf) -> Result<Mat> {
    match imread(image_path.to_str().unwrap(), IMREAD_UNCHANGED) {
        Ok(image) => {
            Ok(image)
        },
        Err(e) => {
            Err(anyhow!("read image failed!"))
        }
    }
}

fn mask_text(image: &Mat) -> Result<Mat> {
    let mut image_hsv: Mat = Default::default();
    let mut image_mask: Mat = Default::default();
    cvt_color(image, &mut image_hsv, COLOR_BGR2HSV, 0)?;
    in_range(&image_hsv, &Scalar::new(0., 0., 0., 0.), &Scalar::new(0., 0., 150., 255.), &mut image_mask)?;
    let kernel = get_structuring_element(MORPH_RECT, Size::new(5, 3), Point::new(-1, -1)).unwrap();
    let mut image_dst: Mat = Default::default();
    dilate(&image_mask, &mut image_dst, &kernel, Point::new(-1, -1), 5, 0, morphology_default_border_value()?).unwrap();
    Ok(image_dst)
}

fn find_textarea_from_mask(image: &Mat) -> Result<Vec<Rect>> {
    let mut contours: VectorOfVectorOfPoint = Default::default();
    let mut rect_result: Vec<Rect> = Default::default();
    find_contours(image, &mut contours, RETR_EXTERNAL, CHAIN_APPROX_NONE, Default::default()).unwrap();
    for contour in contours {
        let rect = bounding_rect(&contour).unwrap();
        if rect.height < rect.width && rect.height > 15 {
            if rect.width / rect.height < 5 && rect.width < (image.cols() / 2) {
                rect_result.push(rect);
            }
        }
    }
    Ok(rect_result)
}

fn scan_image(tess: &mut TessBaseApi, image: &Mat) -> Result<Text> {
    tess.set_image(image.data_bytes().unwrap(),
                   image.cols() as c_int,
                   image.rows() as c_int,
                   image.channels(), (image.cols() * image.channels()) as c_int).expect("TODO: panic message");
    match tess.get_utf8_text() {
        Ok(text) => { Ok(text) },
        Err(e) => { Err(anyhow!("scan image failed!")) }
    }
}

fn find_textarea_from_image(image: &Mat) -> Result<Vec<Rect>> {
    let masked = mask_text(image)?;
    find_textarea_from_mask(&masked)
}

fn init_tess(lang: &CStr) -> Result<TessBaseApi> {
    let mut ocr = TessBaseApi::create();
    match ocr.init_2(None, Some(lang)) {
        Ok(_) => { Ok(ocr) },
        Err(e) => { Err(anyhow!("init tesseract failed!")) }
    }
}

fn mask_my_name(image_path: PathBuf, target_string: String) -> Result<()> {
    let mut image = load_image(image_path)?;
    let mut target_image: Mat = Default::default();
    let lang = CString::new("eng")?;
    match init_tess(lang.as_c_str()) {
        Ok(mut tess) => {
            for area in find_textarea_from_image(&image)? {
                let mut roi = Mat::roi(&image, area).unwrap();
                roi.copy_to(&mut target_image).unwrap();
                match scan_image(&mut tess, &target_image) {
                    Ok(text) => {
                        if text.as_ref().to_str()?.to_lowercase().contains(&target_string.to_lowercase()) {
                            Mat::new_rows_cols_with_default(roi.rows(), roi.cols(), roi.typ(), Scalar::new(0., 0., 0., 255.)).unwrap().copy_to(&mut roi)?;
                            imwrite("output.jpg", &image, &Default::default()).unwrap();
                        }
                    },
                    Err(_) => {

                    }
                }
            }
        },
        Err(e) => {

        }
    }
    Ok(())
}

struct Cli {
    image_path: PathBuf,
    target_string: String
}

fn main() {
    let path = env::args().nth(1).expect("no path given");
    let target_string = env::args().nth(2).unwrap_or("".to_string());
    let args = Cli {
        image_path: PathBuf::from(path),
        target_string,
    };
    mask_my_name(args.image_path, args.target_string).unwrap();
}
