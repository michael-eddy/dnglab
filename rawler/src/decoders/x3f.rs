//! X3F (Foveon) decoder.
//!
//! The TRUE Huffman decoding and the Quattro plane assembly and interpolation
//! are ported from LibRaw (LGPL-2.1/CDDL), see `src/x3f/x3f_utils_patched.cpp`,
//! `src/x3f/x3f_parse_process.cpp` and `internal/x3f_tools.h`.

use std::collections::HashMap;
use std::io::Cursor;

use image::{DynamicImage, ImageFormat};

use crate::RawlerError;
use crate::Result;
use crate::bits::{LEf32, LEu16, LEu32};
use crate::decoders::*;
use crate::exif::Exif;
use crate::formats::tiff::IFD;
use crate::pixarray::PixU16;
use crate::tags::TiffCommonTag;

use super::Camera;
use super::CameraLookup;
use super::image_bytes;

/// Size of the `SECi` image header in front of the pixel data
const X3F_IMAGE_HEADER_SIZE: usize = 28;

/// Size of the `SECc` CAMF header in front of the compressed data
const CAMF_HEADER_SIZE: usize = 28;

/// Offset of the compressed CAMF bitstream behind the Huffman table
const CAMF_DATA_OFFSET: usize = 32;

/// Upper limit for the decoded CAMF data, corrupt sizes must not allocate it
const CAMF_MAX_DECODED: usize = 128 << 20;

/// CAMF entry identifiers, see `X3F_CMbP`, `X3F_CMbT` and `X3F_CMbM` of LibRaw
const CAMF_PROPERTY: u32 = 0x5062_4d43;
const CAMF_TEXT: u32 = 0x5462_4d43;
const CAMF_MATRIX: u32 = 0x4d62_4d43;

/// Format of a JPEG preview (`type` is 2 for previews)
const FORMAT_PREVIEW_JPEG: usize = 0x12;

/// Raw formats that use the TRUE compression, see LibRaw `x3f_tools.h`.
const FORMAT_RAW_TRUE: usize = 0x1e; // X3F_IMAGE_RAW_TRUE / X3F_IMAGE_RAW_MERRILL
const FORMAT_RAW_QUATTRO: usize = 0x23;
const FORMAT_RAW_SDQ: usize = 0x25;
const FORMAT_RAW_SDQH: usize = 0x27;
const FORMAT_RAW_SDQH2: usize = 0x29;

pub fn is_x3f(file: &RawSource) -> bool {
  match file.subview(0, 4) {
    Ok(buf) => buf[0..4] == b"FOVb"[..],
    Err(_) => false,
  }
}

/// Read a little-endian u32 at `pos`, fails if `buf` is too short
fn u32_at(buf: &[u8], pos: usize) -> Result<u32> {
  buf.get(pos..pos + 4).map(|bytes| LEu32(bytes, 0)).ok_or_else(|| behind_eof(pos))
}

/// Read a little-endian u16 at `pos`, fails if `buf` is too short
fn u16_at(buf: &[u8], pos: usize) -> Result<u16> {
  buf.get(pos..pos + 2).map(|bytes| LEu16(bytes, 0)).ok_or_else(|| behind_eof(pos))
}

fn behind_eof(pos: usize) -> RawlerError {
  decoder_error!("X3F: offset {} is behind EOF", pos)
}

#[derive(Debug, Clone)]
struct X3fFile {
  dirs: Vec<X3fDirectory>,
  images: Vec<X3fImage>,
  /// Rotation of the raw image in degrees, 0, 90, 180 or 270
  rotation: u32,
}

#[derive(Debug, Clone)]
struct X3fDirectory {
  offset: usize,
  len: usize,
  id: String,
}

#[derive(Debug, Clone)]
struct X3fImage {
  typ: usize,
  format: usize,
  width: usize,
  height: usize,
  /// Offset of the `SECi` header
  offset: usize,
  /// Length of the `SECi` header and the pixel data
  len: usize,
  /// Offset of the pixel data, behind the `SECi` header
  doffset: usize,
}

impl X3fFile {
  fn new(buf: &[u8]) -> Result<X3fFile> {
    if buf.len() < 12 {
      return Err(decoder_error!("X3F: file is too short"));
    }
    let version = u32_at(buf, 4)?;
    // Version 4 moved the rotation field behind the unique identifier
    let rotation = u32_at(buf, if version >= 0x0004_0000 { 48 } else { 36 })?;

    let dir_offset = u32_at(buf, buf.len() - 4)? as usize;
    if buf.get(dir_offset..dir_offset + 4) != Some(&b"SECd"[..]) {
      return Err(decoder_error!("X3F: directory section is missing"));
    }
    let dir_version = u32_at(buf, dir_offset + 4)?;
    if dir_version < 0x0002_0000 {
      return Err(decoder_error!("X3F: directory version too old {}", dir_version));
    }
    let entries = u32_at(buf, dir_offset + 8)? as usize;
    if entries > 50 {
      return Err(decoder_error!("X3F: directory has too many entries ({})", entries));
    }

    let mut dirs = Vec::with_capacity(entries);
    let mut images = Vec::new();
    for i in 0..entries {
      let dir = X3fDirectory::new(buf, dir_offset + 12 + i * 12)?;
      if dir.id == "IMA2" {
        images.push(X3fImage::new(buf, &dir)?);
      }
      dirs.push(dir);
    }

    Ok(X3fFile { dirs, images, rotation })
  }

  /// The raw image entry of the file
  fn raw_entry(&self) -> Option<&X3fImage> {
    self.images.iter().find(|image| is_raw_image(image.typ, image.format))
  }

  /// The largest JPEG preview of the file
  fn preview_entry(&self) -> Option<&X3fImage> {
    self
      .images
      .iter()
      .filter(|image| image.typ == 2 && image.format == FORMAT_PREVIEW_JPEG)
      .max_by_key(|image| image.width * image.height)
  }

  /// The small JPEG thumbnail of the file, if it has more than one preview
  fn thumbnail_entry(&self) -> Option<&X3fImage> {
    let previews: Vec<&X3fImage> = self
      .images
      .iter()
      .filter(|image| image.typ == 2 && image.format == FORMAT_PREVIEW_JPEG)
      .collect();
    let smallest = previews.iter().copied().min_by_key(|image| image.width * image.height)?;
    (previews.len() > 1).then_some(smallest)
  }

  /// The properties of the `PROP` section, name to value
  fn properties(&self, buf: &[u8]) -> HashMap<String, String> {
    self
      .dirs
      .iter()
      .find(|dir| dir.id == "PROP")
      .and_then(|dir| parse_properties(buf.get(dir.offset..dir.offset + dir.len)?))
      .unwrap_or_default()
  }
}

/// TRUE compressed raw images, `type` is 1 for most of them and 3 for the X530/10 bit formats
fn is_raw_image(typ: usize, format: usize) -> bool {
  match typ {
    1 => matches!(
      format,
      FORMAT_RAW_TRUE | FORMAT_RAW_QUATTRO | FORMAT_RAW_SDQ | FORMAT_RAW_SDQH | FORMAT_RAW_SDQH2
    ),
    // The X530 and 10 bit formats use an unrelated Huffman compression
    3 => format == FORMAT_RAW_TRUE || format == 0x05 || format == 0x06,
    _ => false,
  }
}

fn is_true_format(format: usize) -> bool {
  matches!(
    format,
    FORMAT_RAW_TRUE | FORMAT_RAW_QUATTRO | FORMAT_RAW_SDQ | FORMAT_RAW_SDQH | FORMAT_RAW_SDQH2
  )
}

fn is_quattro_format(format: usize) -> bool {
  matches!(format, FORMAT_RAW_QUATTRO | FORMAT_RAW_SDQ | FORMAT_RAW_SDQH | FORMAT_RAW_SDQH2)
}

impl X3fDirectory {
  fn new(buf: &[u8], offset: usize) -> Result<X3fDirectory> {
    let off = u32_at(buf, offset)? as usize;
    let len = u32_at(buf, offset + 4)? as usize;
    let id = String::from_utf8_lossy(buf.get(offset + 8..offset + 12).ok_or_else(|| behind_eof(offset + 8))?).to_string();
    if off.checked_add(len).is_none_or(|end| end > buf.len()) {
      return Err(RawlerError::decoder_failed("X3F: directory entry exceeds the file"));
    }
    Ok(X3fDirectory { offset: off, len, id })
  }
}

impl X3fImage {
  fn new(buf: &[u8], dir: &X3fDirectory) -> Result<X3fImage> {
    if dir.len < X3F_IMAGE_HEADER_SIZE {
      return Err(decoder_error!("X3F: image entry is too short"));
    }
    let format = u32_at(buf, dir.offset + 12)? as usize;
    Ok(X3fImage {
      typ: u32_at(buf, dir.offset + 8)? as usize,
      format,
      width: u32_at(buf, dir.offset + 16)? as usize,
      height: u32_at(buf, dir.offset + 20)? as usize,
      offset: dir.offset,
      len: dir.len,
      doffset: dir.offset + X3F_IMAGE_HEADER_SIZE,
    })
  }
}

/// Parse the UTF-16 property table of the `PROP` section
fn parse_properties(entry: &[u8]) -> Option<HashMap<String, String>> {
  let num = u32_at(entry, 8).ok()? as usize;
  if num == 0 || num > 1024 {
    return None;
  }
  let table_start: usize = 24;
  let data_start = table_start.checked_add(num.checked_mul(8)?)?;
  let data = entry.get(data_start..)?;
  if !data.len().is_multiple_of(2) {
    return None;
  }
  let mut props = HashMap::with_capacity(num);
  for i in 0..num {
    let pos = table_start + i * 8;
    let name_offset = u32_at(entry, pos).ok()? as usize;
    let value_offset = u32_at(entry, pos + 4).ok()? as usize;
    let name = utf16_string(data, name_offset)?;
    let value = utf16_string(data, value_offset)?;
    props.insert(name, value);
  }
  Some(props)
}

/// A zero terminated UTF-16 string at `offset` UTF-16 units into `data`
fn utf16_string(data: &[u8], offset: usize) -> Option<String> {
  let mut pos = offset.checked_mul(2)?;
  let mut units = Vec::new();
  loop {
    let unit = u16::from_le_bytes(data.get(pos..pos + 2)?.try_into().ok()?);
    pos += 2;
    if unit == 0 {
      break;
    }
    units.push(unit);
    if units.len() > 512 {
      return None;
    }
  }
  Some(String::from_utf16_lossy(&units))
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
  haystack.windows(needle.len()).position(|window| window == needle)
}

/// Read the EXIF data of the APP1 segment of a JPEG preview
fn parse_exif(buf: &[u8], preview: &X3fImage) -> Result<(IFD, Exif)> {
  let end = preview
    .offset
    .checked_add(preview.len)
    .filter(|end| *end <= buf.len())
    .ok_or_else(|| RawlerError::decoder_failed("X3F: preview exceeds the file"))?;
  let jpeg = &buf[preview.doffset..end];
  let exif_pos = find_subslice(jpeg, b"Exif\0\0").ok_or_else(|| RawlerError::decoder_failed("X3F: preview has no EXIF data"))?;
  let base = preview.doffset + exif_pos + 6;
  let base = u32::try_from(base).map_err(|_| RawlerError::decoder_failed("X3F: EXIF offset exceeds 4 GByte"))?;
  let ifd = IFD::new_root(&mut Cursor::new(buf), base)?;
  let exif = Exif::new(&ifd)?;
  Ok((ifd, exif))
}

fn ifd_string(ifd: &IFD, tag: TiffCommonTag) -> Option<String> {
  let value = ifd.get_entry(tag)?.value.as_string()?;
  let value = value.trim_end_matches('\0').trim();
  (!value.is_empty()).then(|| value.to_string())
}

/// Decode a JPEG preview stored in an `IMA2` section
fn jpeg_preview(file: &RawSource, image: &X3fImage) -> Result<DynamicImage> {
  let end = image
    .offset
    .checked_add(image.len)
    .filter(|end| *end <= file.len())
    .ok_or_else(|| RawlerError::decoder_failed("X3F: preview exceeds the file"))?;
  let data = file.subview(image.doffset as u64, (end - image.doffset) as u64)?;
  image::load_from_memory_with_format(data, ImageFormat::Jpeg).map_err(|err| decoder_error!("X3F: failed to decode the JPEG preview: {}", err))
}

/// White balance gains from the CAMF `WhiteBalanceGains` property list.
///
/// The gains of the preset named by the PROP `WB_DESC` are used, with `Auto`
/// as fallback when the file doesn't name a mode. Files without CAMF gains
/// (no CAMF section, an undecodable type or a missing matrix) return `None`;
/// the caller has to estimate the white balance in that case.
fn parse_white_balance(buf: &[u8], dir: &X3fFile, props: &HashMap<String, String>) -> Option<[f32; 4]> {
  let camf = dir.dirs.iter().find(|dir| dir.id == "CAMF")?;
  let entry = buf.get(camf.offset..camf.offset + camf.len)?;
  if entry.len() < CAMF_HEADER_SIZE + CAMF_DATA_OFFSET {
    return None;
  }
  let camf_type = u32_at(entry, 8).ok()?;
  let values = [
    u32_at(entry, 12).ok()?,
    u32_at(entry, 16).ok()?,
    u32_at(entry, 20).ok()?,
    u32_at(entry, 24).ok()?,
  ];
  let decoded = decode_camf(camf_type, values, &entry[CAMF_HEADER_SIZE..])?;
  let entries = camf_entries(&decoded);

  let gains = entries
    .iter()
    .find(|entry| entry.id == CAMF_PROPERTY && entry.name == "WhiteBalanceGains")
    .and_then(camf_properties)?;

  let desc = props.get("WB_DESC").map(|desc| desc.trim()).unwrap_or("");
  let preset = if desc.is_empty() { "Auto" } else { desc };
  let matrix_name = gains.iter().find(|(name, _)| name.eq_ignore_ascii_case(preset)).map(|(_, matrix)| matrix)?;

  let matrix = entries
    .iter()
    .find(|entry| entry.id == CAMF_MATRIX && entry.name == matrix_name.as_str())
    .and_then(camf_matrix3)?;
  normalize_wb_gains(matrix)
}

/// White balance multipliers with the green channel normalized to 1
fn normalize_wb_gains([r, g, b]: [f32; 3]) -> Option<[f32; 4]> {
  if !(r.is_finite() && g.is_finite() && b.is_finite()) || r <= 0.0 || g <= 0.0 || b <= 0.0 {
    return None;
  }
  Some([r / g, 1.0, b / g, f32::NAN])
}

struct CamfEntry<'a> {
  id: u32,
  name: &'a str,
  entry: &'a [u8],
  value_offset: usize,
}

/// Entries of the decoded CAMF data, see `x3f_setup_camf_entries()` of LibRaw
fn camf_entries(decoded: &[u8]) -> Vec<CamfEntry<'_>> {
  let mut entries = Vec::new();
  let mut pos = 0;
  while pos + 20 <= decoded.len() {
    let id = LEu32(decoded, pos);
    if !matches!(id, CAMF_PROPERTY | CAMF_TEXT | CAMF_MATRIX) {
      break;
    }
    let entry_size = LEu32(decoded, pos + 8) as usize;
    let name_offset = LEu32(decoded, pos + 12) as usize;
    let value_offset = LEu32(decoded, pos + 16) as usize;
    if entry_size < 20 || pos + entry_size > decoded.len() || name_offset > value_offset || value_offset > entry_size {
      break;
    }
    let entry = &decoded[pos..pos + entry_size];
    let name = entry[name_offset..value_offset].split(|byte| *byte == 0).next().unwrap_or(&[]);
    let name = std::str::from_utf8(name).unwrap_or("");
    entries.push(CamfEntry { id, name, entry, value_offset });
    pos += entry_size;
  }
  entries
}

/// Properties of a `CMbP` entry, name to value
fn camf_properties(entry: &CamfEntry<'_>) -> Option<HashMap<String, String>> {
  let value = entry.entry.get(entry.value_offset..)?;
  let num = u32_at(value, 0).ok()? as usize;
  let offset = u32_at(value, 4).ok()? as usize;
  if num == 0 || num > 1024 {
    return None;
  }
  let mut props = HashMap::with_capacity(num);
  for i in 0..num {
    let name_offset = offset.checked_add(u32_at(value, 8 + 8 * i).ok()? as usize)?;
    let value_offset = offset.checked_add(u32_at(value, 12 + 8 * i).ok()? as usize)?;
    let name = camf_string(entry.entry, name_offset)?;
    let value = camf_string(entry.entry, value_offset)?;
    props.insert(name, value);
  }
  Some(props)
}

/// The first three values of a float matrix entry, e.g. the `AutoWBGain` entry
fn camf_matrix3(entry: &CamfEntry<'_>) -> Option<[f32; 3]> {
  let value = entry.entry.get(entry.value_offset..)?;
  let matrix_type = u32_at(value, 0).ok()?;
  let dims = u32_at(value, 4).ok()? as usize;
  let data_offset = u32_at(value, 8).ok()? as usize;
  if matrix_type != 3 || dims == 0 || dims > 8 {
    return None;
  }
  let mut elements = 1usize;
  for i in 0..dims {
    elements = elements.checked_mul(u32_at(value, 12 + 12 * i).ok()? as usize)?;
  }
  if elements < 3 {
    return None;
  }
  let data = entry.entry.get(data_offset..)?;
  if data.len() < 12 {
    return None;
  }
  Some([LEf32(data, 0), LEf32(data, 4), LEf32(data, 8)])
}

/// A zero terminated string at `offset` bytes into `data`
fn camf_string(data: &[u8], offset: usize) -> Option<String> {
  let tail = data.get(offset..)?;
  let len = tail.iter().position(|byte| *byte == 0).unwrap_or(tail.len());
  if len == 0 || len > 256 {
    return None;
  }
  Some(String::from_utf8_lossy(&tail[..len]).into_owned())
}

/// Decode the compressed CAMF data, see `x3f_load_camf_decode_type4/5()` of LibRaw
fn decode_camf(camf_type: u32, values: [u32; 4], data: &[u8]) -> Option<Vec<u8>> {
  let mut table = Vec::new();
  let mut pos = 0;
  loop {
    let code_size = *data.get(pos)?;
    let code = *data.get(pos + 1)?;
    pos += 2;
    if code_size == 0 {
      break;
    }
    table.push((code_size, code));
    if table.len() > 255 {
      return None;
    }
  }
  let huffman = TrueHuffman::new(&table).ok()?;
  if data.len() < CAMF_DATA_OFFSET {
    return None;
  }

  let decoded_size = values[0] as usize;
  if decoded_size == 0 || decoded_size > CAMF_MAX_DECODED {
    return None;
  }
  let mut decoded = vec![0u8; decoded_size];

  match camf_type {
    // Merrill cameras: 12 bit values packed into bytes, like TRUE image data
    4 => {
      let bias = values[1] as i32;
      let block_size = values[2] as usize;
      let block_count = values[3] as usize;
      if block_size == 0 || block_count == 0 || block_size > 1_000_000 || block_count > 1_000_000 {
        return None;
      }
      let mut reader = BitReader::new(data, CAMF_DATA_OFFSET);
      let mut row_start_acc = [[bias; 2]; 2];
      let mut dst = 0usize;
      let mut odd = false;
      'outer: for row in 0..block_count {
        let odd_row = row & 1;
        let mut acc = [0i32; 2];
        for col in 0..block_size {
          let odd_col = col & 1;
          let diff = get_true_diff(&mut reader, &huffman).ok()?;
          let prev = if col < 2 { row_start_acc[odd_row][odd_col] } else { acc[odd_col] };
          let value = prev.wrapping_add(diff);
          acc[odd_col] = value;
          if col < 2 {
            row_start_acc[odd_row][odd_col] = value;
          }
          if !odd {
            decoded[dst] = ((value >> 4) & 0xff) as u8;
            dst += 1;
            if dst >= decoded_size {
              break 'outer;
            }
            decoded[dst] = (value.wrapping_shl(4) & 0xf0) as u8;
          } else {
            decoded[dst] |= ((value >> 8) & 0x0f) as u8;
            dst += 1;
            if dst >= decoded_size {
              break 'outer;
            }
            decoded[dst] = (value & 0xff) as u8;
            dst += 1;
            if dst >= decoded_size {
              break 'outer;
            }
          }
          odd = !odd;
        }
      }
    }
    // Quattro cameras: one byte per difference
    5 => {
      let mut reader = BitReader::new(data, CAMF_DATA_OFFSET);
      let mut acc = values[1] as i32;
      for byte in decoded.iter_mut() {
        acc = acc.wrapping_add(get_true_diff(&mut reader, &huffman).ok()?);
        *byte = (acc & 0xff) as u8;
      }
    }
    _ => return None,
  }

  Some(decoded)
}

#[derive(Debug, Clone)]
pub struct X3fDecoder {
  dir: X3fFile,
  raw: X3fImage,
  camera: CameraLookup,
  exif: Exif,
  props: HashMap<String, String>,
  wb: [f32; 4],
}

impl X3fDecoder {
  pub fn new(file: &RawSource, rawloader: &RawLoader) -> Result<X3fDecoder> {
    let buf: &[u8] = &file[..];
    let dir = X3fFile::new(buf)?;
    let raw = dir
      .raw_entry()
      .ok_or_else(|| RawlerError::decoder_failed("X3F: file contains no raw image"))?
      .clone();

    let props = dir.properties(buf);
    let (ifd, exif) = match dir.preview_entry() {
      Some(preview) => match parse_exif(buf, preview) {
        Ok((ifd, exif)) => (Some(ifd), exif),
        Err(err) => {
          log::warn!("X3F: {}", err);
          (None, Exif::default())
        }
      },
      None => (None, Exif::default()),
    };

    // The larger previews carry EXIF, files without it store the camera name in the properties
    let make = ifd
      .as_ref()
      .and_then(|ifd| ifd_string(ifd, TiffCommonTag::Make))
      .or_else(|| props.get("CAMMANUF").cloned())
      .unwrap_or_default();
    let model = ifd
      .as_ref()
      .and_then(|ifd| ifd_string(ifd, TiffCommonTag::Model))
      .or_else(|| props.get("CAMMODEL").cloned())
      .unwrap_or_default();
    if make.is_empty() || model.is_empty() {
      return Err(RawlerError::decoder_failed("X3F: couldn't determine the camera make and model"));
    }

    let camera = rawloader.lookup_camera_with_everything(&make, &model, "");
    let wb = parse_white_balance(buf, &dir, &props).unwrap_or([f32::NAN; 4]);
    if wb[0].is_nan() {
      log::debug!("X3F: no white balance gains in CAMF, white balance is NaN");
    } else {
      log::debug!("X3F: white balance {:?}", wb);
    }
    Ok(X3fDecoder {
      dir,
      raw,
      camera,
      exif,
      props,
      wb,
    })
  }

  fn get_wb(&self) -> [f32; 4] {
    self.wb
  }

  fn orientation(&self) -> Option<Orientation> {
    Orientation::from_degrees(self.dir.rotation as i32)
  }

  fn decode_raw(&self, buf: &[u8], dummy: bool) -> Result<PixU16> {
    let width = self.raw.width;
    let height = self.raw.height;
    let samples = width.checked_mul(3).ok_or_else(|| RawlerError::decoder_failed("X3F: image width overflows"))?;
    image_bytes("X3F", samples, height, 16)?;

    if dummy {
      return Ok(PixU16::new_uninit(samples, height));
    }

    if !is_true_format(self.raw.format) {
      return Err(decoder_error!("X3F: unsupported raw format 0x{:x}", self.raw.format));
    }
    let end = self
      .raw
      .offset
      .checked_add(self.raw.len)
      .filter(|end| *end <= buf.len())
      .ok_or_else(|| RawlerError::decoder_failed("X3F: raw image exceeds the file"))?;
    let data = &buf[self.raw.doffset..end];

    let info = TrueInfo::parse(data, width, height, self.raw.format)?;
    let pixels = decode_true(data, &info, width, height, self.camera.supported()?)?;
    Ok(PixU16::new_with(pixels, samples, height))
  }
}

impl Decoder for X3fDecoder {
  fn raw_image(&self, file: &RawSource, _params: &RawDecodeParams, dummy: bool) -> Result<RawImage> {
    let camera = self.camera.supported()?.clone();
    let image = self.decode_raw(&file[..], dummy)?;
    let cpp = 3;
    let mut raw = RawImage::new(camera, image, cpp, self.get_wb(), RawPhotometricInterpretation::LinearRaw, None, None, dummy);
    raw.orientation = self.orientation();
    Ok(raw)
  }

  fn check_supported(&self) -> Result<()> {
    self.camera.supported().map(|_| ())
  }

  fn format_dump(&self) -> FormatDump {
    todo!()
  }

  fn format_hint(&self) -> FormatHint {
    FormatHint::X3F
  }

  fn raw_metadata(&self, _file: &RawSource, _params: &RawDecodeParams) -> Result<RawMetadata> {
    let mut exif = self.exif.clone();
    if exif.iso_speed_ratings.is_none()
      && let Some(iso) = self.props.get("ISO").and_then(|iso| iso.trim().parse::<u16>().ok())
    {
      exif.iso_speed_ratings = Some(iso);
    }
    Ok(RawMetadata::new(self.camera.metadata(), exif))
  }

  fn thumbnail_image(&self, file: &RawSource, _params: &RawDecodeParams) -> Result<Option<DynamicImage>> {
    self.dir.thumbnail_entry().map(|image| jpeg_preview(file, image)).transpose()
  }

  fn preview_image(&self, file: &RawSource, params: &RawDecodeParams) -> Result<Option<DynamicImage>> {
    if params.image_index != 0 {
      return Ok(None);
    }
    self.dir.preview_entry().map(|image| jpeg_preview(file, image)).transpose()
  }
}

/// Header of a TRUE compressed raw image, planes hold `(columns, rows)`
#[derive(Debug, Clone)]
struct TrueInfo {
  planes: [[usize; 2]; 3],
  quattro: bool,
  quattro_layout: bool,
  seeds: [u16; 3],
  table: Vec<(u8, u8)>,
  /// Start of each plane within the data block
  starts: [usize; 3],
}

impl TrueInfo {
  fn parse(data: &[u8], width: usize, height: usize, format: usize) -> Result<TrueInfo> {
    if width == 0 || height == 0 || width > 65535 || height > 65535 {
      return Err(decoder_error!("X3F: invalid raw image size {}x{}", width, height));
    }
    let quattro = is_quattro_format(format);
    let mut pos = 0usize;
    let mut planes = [[width, height]; 3];
    let mut quattro_layout = false;
    if quattro {
      for plane in planes.iter_mut() {
        *plane = [u16_at(data, pos)? as usize, u16_at(data, pos + 2)? as usize];
        pos += 4;
      }
      if planes[0][1] == height / 2 {
        quattro_layout = true;
      } else if planes[0][1] != height {
        return Err(decoder_error!("X3F: Quattro plane height {} does not fit {}", planes[0][1], height));
      }
    }

    let seeds = [u16_at(data, pos)?, u16_at(data, pos + 2)?, u16_at(data, pos + 4)?];
    pos += 8; // three seeds and one unused value

    let mut table = Vec::new();
    loop {
      let code_size = *data.get(pos).ok_or_else(|| behind_eof(pos))?;
      let code = *data.get(pos + 1).ok_or_else(|| behind_eof(pos + 1))?;
      pos += 2;
      table.push((code_size, code));
      if code_size == 0 {
        break;
      }
      if table.len() > 255 {
        return Err(RawlerError::decoder_failed("X3F: TRUE Huffman table is too large"));
      }
    }

    if quattro {
      u32_at(data, pos)?; // unused
      pos += 4;
    }

    let plane_sizes = [u32_at(data, pos)? as usize, u32_at(data, pos + 4)? as usize, u32_at(data, pos + 8)? as usize];
    pos += 12;

    let mut starts = [0usize; 3];
    let mut offset = pos;
    for (i, size) in plane_sizes.iter().enumerate() {
      starts[i] = offset;
      // Planes start at 16 byte boundaries, see LibRaw x3f_load_true()
      let padded = size.checked_add(15).ok_or_else(|| RawlerError::decoder_failed("X3F: plane size overflows"))? / 16 * 16;
      offset = offset
        .checked_add(padded)
        .ok_or_else(|| RawlerError::decoder_failed("X3F: plane size overflows"))?;
    }
    if offset > data.len() {
      return Err(RawlerError::decoder_failed("X3F: TRUE pixel data is truncated"));
    }

    Ok(TrueInfo {
      planes,
      quattro,
      quattro_layout,
      seeds,
      table,
      starts,
    })
  }
}

/// Bit reader for the TRUE data, the bits of every byte are read from the most significant bit down
struct BitReader<'a> {
  data: &'a [u8],
  pos: usize,
  byte: u8,
  left: u8,
}

impl<'a> BitReader<'a> {
  fn new(data: &'a [u8], start: usize) -> Self {
    Self {
      data,
      pos: start,
      byte: 0,
      left: 0,
    }
  }

  fn bit(&mut self) -> Result<u8> {
    if self.left == 0 {
      self.byte = *self
        .data
        .get(self.pos)
        .ok_or_else(|| RawlerError::decoder_failed("X3F: unexpected end of TRUE data"))?;
      self.pos += 1;
      self.left = 8;
    }
    self.left -= 1;
    Ok((self.byte >> self.left) & 1)
  }
}

/// Canonical Huffman table of the TRUE compression, entry `i` of the file table is the
/// number of bits used by the difference when symbol `i` is decoded
struct TrueHuffman {
  /// `lut[length][code]` is the symbol, `u16::MAX` if the code is unused
  lut: Vec<[u16; 256]>,
}

impl TrueHuffman {
  fn new(table: &[(u8, u8)]) -> Result<Self> {
    let mut lut = vec![[u16::MAX; 256]; 9];
    for (symbol, (length, code)) in table.iter().enumerate() {
      let length = *length as usize;
      if length == 0 {
        continue;
      }
      if length > 8 {
        return Err(decoder_error!("X3F: invalid TRUE Huffman code length {}", length));
      }
      let index = (*code >> (8 - length)) as usize;
      lut[length][index] = symbol as u16;
    }
    Ok(TrueHuffman { lut })
  }

  fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
    let mut code = 0usize;
    for length in 1..=8 {
      code = (code << 1) | reader.bit()? as usize;
      let symbol = self.lut[length][code];
      if symbol != u16::MAX {
        return Ok(symbol);
      }
    }
    Err(RawlerError::decoder_failed("X3F: invalid TRUE Huffman code"))
  }
}

fn get_true_diff(reader: &mut BitReader<'_>, huffman: &TrueHuffman) -> Result<i32> {
  let bits = huffman.decode(reader)? as u32;
  if bits == 0 {
    return Ok(0);
  }
  if bits > 24 {
    return Err(decoder_error!("X3F: invalid TRUE difference width {}", bits));
  }
  let first = reader.bit()? as i32;
  let mut diff = first;
  for _ in 1..bits {
    diff = (diff << 1) + reader.bit()? as i32;
  }
  if first == 0 {
    diff -= (1 << bits) - 1;
  }
  Ok(diff)
}

/// Decode one color plane into `out`, `channel` is the offset of the color within a pixel
#[allow(clippy::too_many_arguments)]
fn decode_plane(
  data: &[u8],
  plane_offset: usize,
  huffman: &TrueHuffman,
  seed: u16,
  rows: usize,
  cols: usize,
  area_rows: usize,
  area_cols: usize,
  channels: usize,
  channel: usize,
  out: &mut [u16],
) -> Result<()> {
  if rows == 0 || cols == 0 || rows != area_rows || cols < area_cols {
    return Err(decoder_error!(
      "X3F: plane of {}x{} does not fit the {}x{} target",
      cols,
      rows,
      area_cols,
      area_rows
    ));
  }
  let mut reader = BitReader::new(data, plane_offset);
  let mut row_start_acc = [[seed as i32; 2]; 2];
  for row in 0..rows {
    let odd_row = row & 1;
    let mut acc = [0i32; 2];
    for col in 0..cols {
      let odd_col = col & 1;
      let diff = get_true_diff(&mut reader, huffman)?;
      let prev = if col < 2 { row_start_acc[odd_row][odd_col] } else { acc[odd_col] };
      let value = prev.wrapping_add(diff);
      acc[odd_col] = value;
      if col < 2 {
        row_start_acc[odd_row][odd_col] = value;
      }
      // Additional data right of the target area is discarded, see true_decode_one_color()
      if col >= area_cols {
        continue;
      }
      out[(row * area_cols + col) * channels + channel] = value as u16;
    }
  }
  Ok(())
}

fn decode_true(data: &[u8], info: &TrueInfo, width: usize, height: usize, camera: &Camera) -> Result<Vec<u16>> {
  let huffman = TrueHuffman::new(&info.table)?;
  let mut out = vec![0u16; width * height * 3];

  if info.quattro && info.quattro_layout {
    let [half_cols, half_rows] = info.planes[0];
    let [top_cols, top_rows] = info.planes[2];
    image_bytes(
      "X3F",
      half_cols
        .checked_mul(3)
        .ok_or_else(|| RawlerError::decoder_failed("X3F: plane width overflows"))?,
      half_rows,
      16,
    )?;
    image_bytes("X3F", top_cols, top_rows, 16)?;
    let mut rgb = vec![0u16; half_cols * half_rows * 3];
    let mut top = vec![0u16; top_cols * top_rows];

    for color in 0..3 {
      if color == 2 {
        decode_plane(
          data,
          info.starts[2],
          &huffman,
          info.seeds[2],
          top_rows,
          top_cols,
          top_rows,
          top_cols,
          1,
          0,
          &mut top,
        )?;
      } else {
        let [cols, rows] = info.planes[color];
        decode_plane(
          data,
          info.starts[color],
          &huffman,
          info.seeds[color],
          rows,
          cols,
          half_rows,
          half_cols,
          3,
          color,
          &mut rgb,
        )?;
      }
    }

    // R and G live in the two half resolution planes, B in the full resolution top plane
    for row in 0..half_rows.min(height / 2) {
      for col in 0..half_cols.min(width / 2) {
        let src = (row * half_cols + col) * 3;
        let dst = ((row * 2) * width + col * 2) * 3;
        out[dst] = rgb[src];
        out[dst + 1] = rgb[src + 1];
      }
    }
    for row in 0..top_rows.min(height) {
      for col in 0..top_cols.min(width) {
        out[(row * width + col) * 3 + 2] = top[row * top_cols + col];
      }
    }
  } else {
    for color in 0..3 {
      let [cols, rows] = info.planes[color];
      decode_plane(
        data,
        info.starts[color],
        &huffman,
        info.seeds[color],
        rows,
        cols,
        height,
        width,
        3,
        color,
        &mut out,
      )?;
    }
  }

  if info.quattro {
    interpolate_quattro(&mut out, width, height, camera, info.quattro_layout);
  }

  Ok(out)
}

/// The autofocus pixel interpolation of the Quattro sensor, sizes and parameters are
/// taken from LibRaw x3f_load_raw()
fn interpolate_quattro(img: &mut [u16], width: usize, height: usize, camera: &Camera, quattro_layout: bool) {
  let [left, top, right, bottom] = camera.crop_area.unwrap_or([0, 0, 0, 0]);
  let crop_width = width.saturating_sub(left + right);
  let crop_height = height.saturating_sub(top + bottom);
  let black = camera.blacklevel.as_ref().and_then(|levels| levels.first()).copied().unwrap_or(0) as i32;

  match (width, height) {
    (5888, 3672) => interpolate_af(img, width, height, left, top, crop_width, crop_height, 32, 8, 2, black),
    (5888, 3776) => interpolate_af_sd(img, width, height, left, top, crop_width, crop_height, 216, 464, width - 1, 3312, 16, 32, 2),
    (6656, 4480) => interpolate_af_sd(img, width, height, left, top, crop_width, crop_height, 232, 592, width - 1, 3920, 16, 32, 2),
    (3328, 2240) => interpolate_af_sd(img, width, height, left, top, crop_width, crop_height, 116, 296, width - 1, 2200, 8, 16, 1),
    (5504, 3680) => interpolate_af_sd(img, width, height, left, top, crop_width, crop_height, 8, 192, width - 1, 3185, 16, 32, 2),
    (2752, 1840) => interpolate_af_sd(img, width, height, left, top, crop_width, crop_height, 4, 96, width - 1, 1800, 8, 16, 1),
    (2944, 1836) => interpolate_af(img, width, height, left, top, crop_width, crop_height, 16, 4, 1, black),
    (2944, 1888) => interpolate_af_sd(img, width, height, left, top, crop_width, crop_height, 108, 232, width - 1, 1656, 8, 16, 1),
    _ => {}
  }

  if quattro_layout {
    interpolate_rg(img, width, height);
  }
}

/// Fill the R and G channels that the half resolution planes don't cover, see
/// LibRaw x3f_dpq_interpolate_rg()
fn interpolate_rg(img: &mut [u16], width: usize, height: usize) {
  let half_width = width / 2;
  let half_height = height / 2;
  let stride = width * 3;
  for color in 0..2 {
    for y in 2..half_height.saturating_sub(2) {
      let mut row0 = (y * 2) * stride + color;
      let mut row1 = (y * 2 + 1) * stride + color;
      for _ in 2..half_width.saturating_sub(2) {
        let value = img[row0];
        img[row0 + 3] = value;
        img[row1] = value;
        img[row1 + 3] = value;
        row0 += 6;
        row1 += 6;
      }
    }
  }
}

/// LibRaw x3f_dpq_interpolate_af_sd()
#[allow(clippy::too_many_arguments)]
fn interpolate_af_sd(
  img: &mut [u16],
  width: usize,
  height: usize,
  left: usize,
  top: usize,
  crop_width: usize,
  crop_height: usize,
  xstart: usize,
  ystart: usize,
  xend: usize,
  yend: usize,
  xstep: usize,
  ystep: usize,
  scale: usize,
) {
  let stride = width * 3;
  let mut y = ystart;
  while y <= yend && y < crop_height + top {
    if y < scale || y + scale >= height {
      y += ystep;
      continue;
    }
    let row0 = y * stride;
    let row1 = (y + 1) * stride;
    let row_minus = (y - scale) * stride;
    let row_plus = (y + scale) * stride;
    let row_minus1 = (y - 1) * stride;
    let mut x = xstart;
    while x < xend && x < crop_width + left {
      if x + scale >= width {
        break;
      }
      let steps = [x - scale, x, x + scale];
      let mut sum_r = 0.0f32;
      let mut sum_g = 0.0f32;
      for (i, xi) in steps.iter().enumerate() {
        sum_r += img[row_minus + xi * 3] as f32 + img[row_plus + xi * 3] as f32;
        sum_g += img[row_minus + xi * 3 + 1] as f32 + img[row_plus + xi * 3 + 1] as f32;
        if i != 1 {
          sum_r += img[row0 + xi * 3] as f32;
          sum_g += img[row0 + xi * 3 + 1] as f32;
        }
      }
      img[row0 + x * 3] = (sum_r / 8.0) as u16;
      img[row0 + x * 3 + 1] = (sum_g / 8.0) as u16;

      if scale == 2 {
        let pixel0 = row0 + x * 3 + 3;
        let pixel1 = row1 + x * 3 + 3;
        let mut sum_g0 = 0.0f32;
        let mut sum_g1 = 0.0f32;
        let mut count = 0.0f32;
        for (i, xi) in steps.iter().enumerate() {
          sum_g0 += img[row_minus1 + xi * 3 + 2] as f32;
          sum_g1 += img[row_plus + xi * 3 + 2] as f32;
          count += 1.0;
          if i != 1 {
            sum_g0 += img[row0 + xi * 3 + 2] as f32;
            sum_g1 += img[row1 + xi * 3 + 2] as f32;
            count += 1.0;
          }
        }
        if count > 1.0 {
          img[pixel0 + 2] = (sum_g0 / count) as u16;
          img[pixel1 + 2] = (sum_g1 / count) as u16;
        }
      }
      x += xstep;
    }
    y += ystep;
  }
}

/// LibRaw x3f_dpq_interpolate_af()
#[allow(clippy::too_many_arguments)]
fn interpolate_af(
  img: &mut [u16],
  width: usize,
  height: usize,
  left: usize,
  top: usize,
  crop_width: usize,
  crop_height: usize,
  xstep: usize,
  ystep: usize,
  scale: usize,
  black: i32,
) {
  let stride = width * 3;
  let clip = |value: f32| value.min(16383.0) as u16;
  let mut y = 0;
  while y < crop_height + top {
    if y < top || y < scale || y > height.saturating_sub(scale) {
      y += ystep;
      continue;
    }
    let row0 = y * stride;
    let row_minus = (y - scale) * stride;
    let row_plus = (y + scale) * stride;
    let mut x = 0;
    while x < crop_width + left {
      if x < left || x < scale || x > width.saturating_sub(scale) {
        x += xstep;
        continue;
      }
      let pixel0 = row0 + x * 3;
      let candidates = [row_minus + x * 3, row_plus + x * 3, row0 + (x - scale) * 3, row0 + (x + scale) * 3];
      let mut nearest = candidates[0];
      for &candidate in &candidates[1..] {
        if (img[candidate + 2] as i32 - img[pixel0 + 2] as i32).abs() > (img[nearest + 2] as i32 - img[pixel0 + 2] as i32).abs() {
          nearest = candidate;
        }
      }
      let blocal = img[pixel0 + 2] as i32;
      let bnear = img[nearest + 2] as i32;
      if blocal < black + 16 || bnear < black + 16 {
        for channel in 0..2 {
          let value = (img[pixel0 + channel] as i32).max(black);
          img[pixel0 + channel] = clip(((value - black) * 4 + black) as f32);
        }
      } else {
        let multip = (bnear - black) as f32 / (blocal - black) as f32;
        let bright = (img[nearest] as f32).max(black as f32);
        let bright1 = (img[nearest + 1] as f32).max(black as f32);
        for (channel, bright) in [bright, bright1].into_iter().enumerate() {
          let value = (img[pixel0 + channel] as i32).max(black);
          let own = (value - black) as f32 * 3.75 + black as f32;
          let other = (bright - black as f32) * multip + black as f32;
          img[pixel0 + channel] = clip((other + own) / 2.0);
        }
      }
      x += xstep;
    }
    y += ystep;
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn utf16_property_is_read() {
    let mut data = vec![b'S', 0, b'I', 0, b'S', 0, b'O', 0, 0, 0];
    data.extend_from_slice(&[b'2', 0, b'0', 0, b'0', 0, 0, 0]);
    assert_eq!(utf16_string(&data, 0).as_deref(), Some("SISO"));
    assert_eq!(utf16_string(&data, 5).as_deref(), Some("200"));
    assert_eq!(utf16_string(&data, 100), None);
  }

  #[test]
  fn true_huffman_codes_are_decoded() {
    // Symbol 0 has the code 0, symbol 1 the code 1, both one bit long
    let huffman = TrueHuffman::new(&[(1, 0x00), (1, 0x80), (0, 0)]).expect("table");
    let data = [0b0100_0000u8];
    let mut reader = BitReader::new(&data, 0);
    assert_eq!(huffman.decode(&mut reader).ok(), Some(0));
    assert_eq!(huffman.decode(&mut reader).ok(), Some(1));
  }

  #[test]
  fn find_exif_marker() {
    assert_eq!(find_subslice(b"\xff\xd8\xff\xe1Exif\x00\x00II", b"Exif\x00\x00"), Some(4));
    assert_eq!(find_subslice(b"\xff\xd8", b"Exif\x00\x00"), None);
  }

  /// An entry of the decoded CAMF data: header, zero terminated name and value
  fn camf_entry(id: u32, name: &str, value: &[u8]) -> Vec<u8> {
    let value_offset = 20 + name.len() + 1;
    let entry_size = value_offset + value.len();
    let mut entry = Vec::with_capacity(entry_size);
    entry.extend_from_slice(&id.to_le_bytes());
    entry.extend_from_slice(&0x0001_0000u32.to_le_bytes());
    entry.extend_from_slice(&(entry_size as u32).to_le_bytes());
    entry.extend_from_slice(&20u32.to_le_bytes());
    entry.extend_from_slice(&(value_offset as u32).to_le_bytes());
    entry.extend_from_slice(name.as_bytes());
    entry.push(0);
    entry.extend_from_slice(value);
    entry
  }

  #[test]
  fn camf_white_balance_gains_are_read() {
    let preset_name = "WhiteBalanceGains";
    let strings_start = 20 + preset_name.len() + 1 + 16;
    let mut props = Vec::new();
    props.extend_from_slice(&1u32.to_le_bytes());
    props.extend_from_slice(&(strings_start as u32).to_le_bytes());
    props.extend_from_slice(&0u32.to_le_bytes());
    props.extend_from_slice(&5u32.to_le_bytes());
    props.extend_from_slice(b"Auto\0AutoWBGain\0");

    let matrix_name = "AutoWBGain";
    let data_offset = 20 + matrix_name.len() + 1 + 24;
    let mut matrix = Vec::new();
    matrix.extend_from_slice(&3u32.to_le_bytes()); // float values
    matrix.extend_from_slice(&1u32.to_le_bytes()); // one dimension
    matrix.extend_from_slice(&(data_offset as u32).to_le_bytes());
    matrix.extend_from_slice(&3u32.to_le_bytes()); // three elements
    matrix.extend_from_slice(&0u32.to_le_bytes());
    matrix.extend_from_slice(&0u32.to_le_bytes());
    for value in [2.0f32, 1.0, 0.5] {
      matrix.extend_from_slice(&value.to_le_bytes());
    }

    let mut decoded = camf_entry(CAMF_PROPERTY, preset_name, &props);
    decoded.extend_from_slice(&camf_entry(CAMF_MATRIX, matrix_name, &matrix));

    let entries = camf_entries(&decoded);
    assert_eq!(entries.len(), 2);
    let props = camf_properties(&entries[0]).expect("properties");
    assert_eq!(props.get("Auto").map(String::as_str), Some("AutoWBGain"));
    assert_eq!(camf_matrix3(&entries[1]), Some([2.0, 1.0, 0.5]));

    let wb = normalize_wb_gains([2.0, 1.0, 0.5]).expect("gains");
    assert_eq!(&wb[..3], &[2.0, 1.0, 0.5]);
    assert!(wb[3].is_nan());

    // The green channel is not always 1
    let wb = normalize_wb_gains([0.9573, 1.05, 1.2267]).expect("gains");
    assert!((wb[0] - 0.9117).abs() < 1e-3);
    assert_eq!(wb[1], 1.0);
    assert!((wb[2] - 1.1683).abs() < 1e-3);

    assert!(normalize_wb_gains([0.0, 1.0, 1.0]).is_none());
    assert!(normalize_wb_gains([f32::NAN, 1.0, 1.0]).is_none());
  }

  /// A `SECi` section whose planes decode to their seed values: the Huffman
  /// table contains only the zero difference symbol, so one bit is read per
  /// pixel and the compressed plane is all zero bits.
  fn sec_i(width: usize, height: usize, format: usize, planes: [[usize; 2]; 3], seeds: [u16; 3]) -> Vec<u8> {
    let mut section = Vec::new();
    section.extend_from_slice(b"SECi");
    section.extend_from_slice(&0x0002_0000u32.to_le_bytes());
    section.extend_from_slice(&1u32.to_le_bytes()); // type, raw
    section.extend_from_slice(&(format as u32).to_le_bytes());
    section.extend_from_slice(&(width as u32).to_le_bytes());
    section.extend_from_slice(&(height as u32).to_le_bytes());
    section.extend_from_slice(&0u32.to_le_bytes()); // row stride, set behind

    if is_quattro_format(format) {
      for [cols, rows] in planes {
        section.extend_from_slice(&(cols as u16).to_le_bytes());
        section.extend_from_slice(&(rows as u16).to_le_bytes());
      }
    }
    for seed in seeds {
      section.extend_from_slice(&seed.to_le_bytes());
    }
    section.extend_from_slice(&0u16.to_le_bytes()); // unused
    // Symbol 0 is the one bit code 0 with a zero width difference
    section.extend_from_slice(&[1, 0x00, 0x00, 0x00]);
    if is_quattro_format(format) {
      section.extend_from_slice(&0u32.to_le_bytes()); // unused
    }

    let plane_bytes: Vec<usize> = planes.iter().map(|[cols, rows]| cols * rows).map(|bits| bits.div_ceil(8)).collect();
    for bytes in &plane_bytes {
      section.extend_from_slice(&(*bytes as u32).to_le_bytes());
    }

    // Plane i starts at the end of the previous plane, padded to 16 bytes
    let header_len = section.len() - X3F_IMAGE_HEADER_SIZE;
    let padded: Vec<usize> = plane_bytes.iter().map(|bytes| bytes.div_ceil(16) * 16).collect();
    let mut start = header_len;
    for bytes in &padded {
      start += bytes;
    }
    section.resize(X3F_IMAGE_HEADER_SIZE + start, 0);
    let pitch = (section.len() - X3F_IMAGE_HEADER_SIZE) as u32;
    section[24..28].copy_from_slice(&pitch.to_le_bytes());
    section
  }

  /// A `SECp` section with UTF-16 properties
  fn sec_p(props: &[(&str, &str)]) -> Vec<u8> {
    let mut strings: Vec<u16> = Vec::new();
    let mut table = Vec::new();
    for (name, value) in props {
      let name_offset = strings.len();
      strings.extend(name.encode_utf16());
      strings.push(0);
      let value_offset = strings.len();
      strings.extend(value.encode_utf16());
      strings.push(0);
      table.push((name_offset as u32, value_offset as u32));
    }

    let mut section = Vec::new();
    section.extend_from_slice(b"SECp");
    section.extend_from_slice(&0x0002_0000u32.to_le_bytes());
    section.extend_from_slice(&(props.len() as u32).to_le_bytes());
    section.extend_from_slice(&0u32.to_le_bytes()); // character format
    section.extend_from_slice(&0u32.to_le_bytes()); // reserved
    section.extend_from_slice(&0u32.to_le_bytes()); // total length
    for (name_offset, value_offset) in table {
      section.extend_from_slice(&name_offset.to_le_bytes());
      section.extend_from_slice(&value_offset.to_le_bytes());
    }
    for unit in strings {
      section.extend_from_slice(&unit.to_le_bytes());
    }
    section
  }

  /// A minimal X3F file with the given sections and their directory identifiers
  fn x3f_file(rotation: u32, sections: &[(Vec<u8>, [u8; 4])]) -> Vec<u8> {
    let mut file = Vec::new();
    file.extend_from_slice(b"FOVb");
    file.extend_from_slice(&0x0003_0000u32.to_le_bytes());
    file.extend_from_slice(&[0u8; 16]); // unique identifier
    file.extend_from_slice(&0u32.to_le_bytes()); // mark bits
    file.extend_from_slice(&0u32.to_le_bytes()); // columns
    file.extend_from_slice(&0u32.to_le_bytes()); // rows
    file.extend_from_slice(&rotation.to_le_bytes());
    assert_eq!(file.len(), 40);

    let mut entries = Vec::new();
    for (section, id) in sections {
      let offset = file.len() as u32;
      file.extend_from_slice(section);
      entries.push((offset, section.len() as u32, *id));
    }

    let dir_offset = file.len() as u32;
    file.extend_from_slice(b"SECd");
    file.extend_from_slice(&0x0002_0000u32.to_le_bytes());
    file.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (offset, len, id) in entries {
      file.extend_from_slice(&offset.to_le_bytes());
      file.extend_from_slice(&len.to_le_bytes());
      file.extend_from_slice(&id);
    }
    file.extend_from_slice(&dir_offset.to_le_bytes());
    file
  }

  /// A `SECi` section with a JPEG preview
  fn sec_jpeg(width: u32, height: u32, jpeg: Vec<u8>) -> Vec<u8> {
    let mut section = Vec::new();
    section.extend_from_slice(b"SECi");
    section.extend_from_slice(&0x0002_0000u32.to_le_bytes());
    section.extend_from_slice(&2u32.to_le_bytes()); // preview
    section.extend_from_slice(&(FORMAT_PREVIEW_JPEG as u32).to_le_bytes());
    section.extend_from_slice(&width.to_le_bytes());
    section.extend_from_slice(&height.to_le_bytes());
    section.extend_from_slice(&(jpeg.len() as u32).to_le_bytes());
    section.extend_from_slice(&jpeg);
    section
  }

  fn jpeg_bytes(width: u32, height: u32) -> Vec<u8> {
    let image = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(width, height, image::Rgb([10, 20, 30])));
    let mut bytes = Vec::new();
    image.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Jpeg).unwrap();
    bytes
  }

  fn sigma_props(model: &str) -> Vec<u8> {
    sec_p(&[("CAMMANUF", "SIGMA"), ("CAMMODEL", model)])
  }

  #[test]
  fn decodes_a_synthetic_merrill_image() -> Result<()> {
    let (width, height) = (8, 8);
    let raw = sec_i(width, height, FORMAT_RAW_TRUE, [[width, height]; 3], [512, 512, 512]);
    let file = x3f_file(0, &[(raw, *b"IMA2"), (sigma_props("SIGMA DP2 Merrill"), *b"PROP")]);
    let source = RawSource::new_from_slice(&file);

    let decoder = crate::get_decoder(&source)?;
    let image = decoder.raw_image(&source, &RawDecodeParams::default(), false)?;
    assert_eq!((image.width, image.height, image.cpp), (width, height, 3));
    assert_eq!(image.bps, 12);
    assert!(matches!(image.photometric, RawPhotometricInterpretation::LinearRaw));
    assert!(image.wb_coeffs.iter().all(|coeff| coeff.is_nan()));
    assert_eq!(image.orientation, Some(Orientation::Normal));

    let RawImageData::Integer(pixels) = image.data else {
      panic!("X3F must decode to integer data");
    };
    assert_eq!(pixels.len(), width * height * 3);
    assert!(pixels.iter().all(|&pixel| pixel == 512));
    Ok(())
  }

  #[test]
  fn assembles_a_synthetic_quattro_image() -> Result<()> {
    let (width, height) = (16, 16);
    let raw = sec_i(width, height, FORMAT_RAW_SDQH, [[8, 8], [8, 8], [16, 16]], [100, 200, 300]);
    let file = x3f_file(0, &[(raw, *b"IMA2"), (sigma_props("sd Quattro H"), *b"PROP")]);
    let source = RawSource::new_from_slice(&file);

    let decoder = crate::get_decoder(&source)?;
    let image = decoder.raw_image(&source, &RawDecodeParams::default(), false)?;
    assert_eq!((image.width, image.height, image.cpp), (width, height, 3));
    let RawImageData::Integer(pixels) = image.data else {
      panic!("X3F must decode to integer data");
    };
    let pixel = |x: usize, y: usize| {
      let i = (y * width + x) * 3;
      [pixels[i], pixels[i + 1], pixels[i + 2]]
    };

    // R and G are stored at even positions and copied to the odd positions
    // inside the interpolated area, B is full resolution everywhere
    for (x, y) in [(4, 4), (5, 4), (4, 5), (5, 5)] {
      assert_eq!(pixel(x, y), [100, 200, 300]);
    }
    for y in 0..height {
      for x in 0..width {
        assert_eq!(pixel(x, y)[2], 300);
      }
    }
    // The outer two rows and columns are left for the default crop
    assert_eq!(pixel(0, 0), [100, 200, 300]);
    assert_eq!(pixel(1, 0), [0, 0, 300]);
    Ok(())
  }

  #[test]
  fn truncated_true_data_is_an_error() {
    let mut raw = sec_i(8, 8, FORMAT_RAW_TRUE, [[8, 8]; 3], [512, 512, 512]);
    // First plane size: SECi header, seeds and Huffman table
    let size_offset = X3F_IMAGE_HEADER_SIZE + 8 + 4;
    raw[size_offset..size_offset + 4].copy_from_slice(&1_000_000u32.to_le_bytes());
    let file = x3f_file(0, &[(raw, *b"IMA2"), (sigma_props("SIGMA DP2 Merrill"), *b"PROP")]);
    let source = RawSource::new_from_slice(&file);
    let decoder = crate::get_decoder(&source).unwrap();
    let err = decoder.raw_image(&source, &RawDecodeParams::default(), false).unwrap_err();
    assert!(err.to_string().contains("truncated"), "unexpected error: {}", err);
  }

  #[test]
  fn reads_the_embedded_jpeg_previews() -> Result<()> {
    let raw = sec_i(8, 8, FORMAT_RAW_TRUE, [[8, 8]; 3], [512, 512, 512]);
    let file = x3f_file(
      0,
      &[
        (raw.clone(), *b"IMA2"),
        (sec_jpeg(2, 2, jpeg_bytes(2, 2)), *b"IMA2"),
        (sec_jpeg(4, 3, jpeg_bytes(4, 3)), *b"IMA2"),
        (sigma_props("SIGMA DP2 Merrill"), *b"PROP"),
      ],
    );
    let source = RawSource::new_from_slice(&file);
    let decoder = crate::get_decoder(&source)?;

    let preview = decoder.preview_image(&source, &RawDecodeParams::default())?.expect("preview");
    assert_eq!((preview.width(), preview.height()), (4, 3));
    let thumbnail = decoder.thumbnail_image(&source, &RawDecodeParams::default())?.expect("thumbnail");
    assert_eq!((thumbnail.width(), thumbnail.height()), (2, 2));

    // With a single preview there is no separate thumbnail
    let file = x3f_file(
      0,
      &[
        (raw, *b"IMA2"),
        (sec_jpeg(4, 3, jpeg_bytes(4, 3)), *b"IMA2"),
        (sigma_props("SIGMA DP2 Merrill"), *b"PROP"),
      ],
    );
    let source = RawSource::new_from_slice(&file);
    let decoder = crate::get_decoder(&source)?;
    assert!(decoder.preview_image(&source, &RawDecodeParams::default())?.is_some());
    assert!(decoder.thumbnail_image(&source, &RawDecodeParams::default())?.is_none());
    Ok(())
  }
}
