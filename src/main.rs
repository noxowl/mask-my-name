use std::env;
use std::path::PathBuf;
use std::ffi::{CStr, CString, c_int};
use thiserror::Error;
use opencv::core::{in_range, Point, Rect, Scalar, Size};
use opencv::prelude::*;
use opencv::imgcodecs::{imread, IMREAD_UNCHANGED, imwrite};
use opencv::imgproc::{COLOR_BGR2HSV, cvt_color, dilate, MORPH_RECT, get_structuring_element, morphology_default_border_value, find_contours, RETR_EXTERNAL, bounding_rect, CHAIN_APPROX_NONE};
use opencv::types::VectorOfVectorOfPoint;
use tesseract_plumbing::{TessBaseApi, Text};

#[derive(Debug, Error, PartialEq)]
pub enum MaskMyNameError {
    #[error("Read image failed.")]
    ImageReadError(),
    #[error("Mask text failed.")]
    MaskTextError(),
    #[error("Failed to initialize Tesseract.")]
    TessInitError(),
    #[error("Failed to get text from Tesseract.")]
    TessGetTextError(),
    #[error("Failed to create black bar Mat.")]
    MaskingBarCreationError(),
    #[error("No matching string found.")]
    NoMatchingString(),
}

fn load_image(image_path: &PathBuf) -> Result<Mat, MaskMyNameError> {
    match imread(image_path.to_str().expect("failed to convert PathBuf to str."), IMREAD_UNCHANGED) {
        Ok(image) => {
            Ok(image)
        },
        Err(_) => {
            Err(MaskMyNameError::ImageReadError())
        }
    }
}

fn iterations(frame_height: i32) -> i32 {
    if frame_height >= 720 {
        5
    } else {
        3
    }
}

fn max_range(frame_height: i32) -> f64 {
    if frame_height >= 720 {
        30.
    } else {
        80.
    }
}

fn mask_text(image: &Mat) -> Result<Mat, MaskMyNameError> {
    let mut image_hsv: Mat = Default::default();
    let mut image_mask: Mat = Default::default();
    cvt_color(image, &mut image_hsv, COLOR_BGR2HSV, 0).expect("Convert BGR2HSV failed. check input image file.");
    in_range(&image_hsv,
             &Scalar::new(0., 0., 0., 0.),
             &Scalar::new(0., 0., max_range(image.rows()), 255.),
             &mut image_mask).expect("in_range failed. check converted Mat is in HSV Colour space.");
    let kernel = get_structuring_element(MORPH_RECT, Size::new(5, 3), Point::new(-1, -1)).expect("failed to get_structuring_element.");
    let mut image_dst: Mat = Default::default();
    dilate(&image_mask, &mut image_dst, &kernel, Point::new(-1, -1), iterations(image.rows()), 0,
           morphology_default_border_value().expect("Failed to calling morphology_default_border_value.")).expect("Failed to calling dilate.");
    Ok(image_dst)
}

fn find_textarea_from_mask(image: &Mat) -> Result<Vec<Rect>, MaskMyNameError> {
    let mut contours: VectorOfVectorOfPoint = Default::default();
    let mut rect_result: Vec<Rect> = Default::default();
    find_contours(image, &mut contours, RETR_EXTERNAL, CHAIN_APPROX_NONE, Default::default()).expect("find_contours failed.");
    for contour in contours {
        let rect = bounding_rect(&contour).expect("");
        if rect.height < rect.width && rect.height > (image.rows() / 72) {
            if rect.width / rect.height < 15 && rect.width < (image.cols() / 2) {
                rect_result.push(rect);
            }
        }
    }
    Ok(rect_result)
}

fn scan_image(tess: &mut TessBaseApi, image: &Mat) -> Result<Text, MaskMyNameError> {
    tess.set_image(image.data_bytes().expect("Failed to get data_bytes from image Mat."),
                   image.cols() as c_int,
                   image.rows() as c_int,
                   image.channels(), (image.cols() * image.channels()) as c_int).expect("Set image to Tesseract failed.");
    match tess.get_utf8_text() {
        Ok(text) => { Ok(text) },
        Err(_) => { Err(MaskMyNameError::TessGetTextError()) }
    }
}

fn masking_bar(roi: &Mat) -> Result<Mat, MaskMyNameError> {
    match Mat::new_rows_cols_with_default(roi.rows(), roi.cols(), roi.typ(), Scalar::new(255., 255., 255., 255.)) {
        Ok(mat) => { Ok(mat) }
        Err(_) => { Err(MaskMyNameError::MaskingBarCreationError()) }
    }
}

fn init_tess(lang: &CStr) -> Result<TessBaseApi, MaskMyNameError> {
    let mut ocr = TessBaseApi::create();
    match ocr.init_2(None, Some(lang)) {
        Ok(_) => { Ok(ocr) },
        Err(_) => { Err(MaskMyNameError::TessInitError()) }
    }
}

fn supplement_target_string(target: &String) -> Vec<String> {
    let mut strings = Vec::new();
    match target.contains("_") {
        true => {
            strings.push(target.to_lowercase());
            strings.push(target.replace("_", " ").to_lowercase());
        }
        false => {
            strings.push(target.to_lowercase());
        }
    }
    strings
}

fn mask_my_name(lang: CString, image_path: &PathBuf, target_string: &String) -> Result<Mat, MaskMyNameError> {
    let mut success = false;
    let image = load_image(image_path).expect("Can't load image file from path.");
    let mut target_image: Mat = Default::default();
    let strings = supplement_target_string(target_string);
    match init_tess(lang.as_c_str()) {
        Ok(mut tess) => {
            for area in find_textarea_from_mask(&mask_text(&image)?)? {
                let mut roi = Mat::roi(&image, area).expect("Failed to create ROI.");
                roi.copy_to(&mut target_image).expect("Failed to copy roi data.");
                match scan_image(&mut tess, &target_image) {
                    Ok(text) => {
                        let picked = text.as_ref().to_str().unwrap_or("").to_lowercase().replace(".", "").replace(",", "");
                        if strings.iter().any(|s| picked.contains(s)) {
                            success = true;
                            masking_bar(&roi)?.copy_to(&mut roi).expect("Failed to copy black bar data.");
                        }
                    },
                    Err(e) => { return Err(e); }
                }
            }
            match success {
                true => { Ok(image) },
                false => { Err(MaskMyNameError::NoMatchingString()) }
            }
        },
        Err(e) => { Err(e) }
    }
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
    if !&args.image_path.is_file() {
        panic!("Image file not exist! check your input filename or path.")
    }
    match mask_my_name(CString::new("eng").expect("Convert str to CString failed."), &args.image_path, &args.target_string) {
        Ok(image) => {
            println!("Matching found. write masked image to disk.");
            imwrite(format!("{}_masked.{}",
                            &args.image_path.file_stem().unwrap_or("output".as_ref()).to_str().unwrap_or("output"),
                            &args.image_path.extension().unwrap_or("jpg".as_ref()).to_str().unwrap_or("jpg")).as_str(),
                    &image, &Default::default()).expect("Failed to write image data.");
        },
        Err(e) => {
            match e {
                MaskMyNameError::NoMatchingString() => {
                    // TODO: switch to japanese string
                    println!("{}", e);
                }
                _ => {
                    println!("{}", e);
                }
            }
        }
    }
}
