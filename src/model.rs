use std::fs;
use std::path::Path;

pub const STATE_DIM: usize = 8;
pub const ANCHOR_DIM: usize = 4;
pub const INPUT_DIM: usize = STATE_DIM * 2 + ANCHOR_DIM;
pub const HIDDEN_DIM: usize = 32;

const MAGIC: [u8; 4] = *b"NCA1";
const HEADER_LEN: usize = 20;

#[derive(Clone)]
pub struct NetworkParameters {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl NetworkParameters {
    #[allow(dead_code)]
    pub fn load_or_default(path: &Path) -> Self {
        match Self::load(path) {
            Ok(weights) => {
                log::info!("loaded CellForge weights from {}", path.display());
                weights
            }
            Err(err) => {
                log::warn!(
                    "failed to load CellForge weights from {}: {}. falling back to seeded defaults",
                    path.display(),
                    err
                );
                Self::default_seeded()
            }
        }
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|err| format!("unable to read weights file {}: {err}", path.display()))?;

        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {

        if bytes.len() < HEADER_LEN {
            return Err("weights file is too small".to_owned());
        }
        if bytes[0..4] != MAGIC {
            return Err("invalid weights magic header".to_owned());
        }

        let state_dim = read_u32(&bytes, 4)? as usize;
        let input_dim = read_u32(&bytes, 8)? as usize;
        let hidden_dim = read_u32(&bytes, 12)? as usize;

        if state_dim != STATE_DIM || input_dim != INPUT_DIM || hidden_dim != HIDDEN_DIM {
            return Err(format!(
                "dimension mismatch: file has state/input/hidden = {state_dim}/{input_dim}/{hidden_dim}, expected {STATE_DIM}/{INPUT_DIM}/{HIDDEN_DIM}"
            ));
        }

        let mut cursor = HEADER_LEN;
        let w1 = read_f32_vec(&bytes, &mut cursor, INPUT_DIM * HIDDEN_DIM)?;
        let b1 = read_f32_vec(&bytes, &mut cursor, HIDDEN_DIM)?;
        let w2 = read_f32_vec(&bytes, &mut cursor, STATE_DIM * HIDDEN_DIM)?;
        let b2 = read_f32_vec(&bytes, &mut cursor, STATE_DIM)?;

        if cursor != bytes.len() {
            log::warn!("weights payload has {} trailing bytes", bytes.len() - cursor);
        }

        let model = Self { w1, b1, w2, b2 };
        model.validate()?;
        Ok(model)
    }

    #[allow(dead_code)]
    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.validate()?;

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|err| {
                    format!("unable to create directory {}: {err}", parent.display())
                })?;
            }
        }

        let mut bytes = Vec::<u8>::with_capacity(
            HEADER_LEN + (self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()) * 4,
        );
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&(STATE_DIM as u32).to_le_bytes());
        bytes.extend_from_slice(&(INPUT_DIM as u32).to_le_bytes());
        bytes.extend_from_slice(&(HIDDEN_DIM as u32).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        append_f32_slice(&mut bytes, &self.w1);
        append_f32_slice(&mut bytes, &self.b1);
        append_f32_slice(&mut bytes, &self.w2);
        append_f32_slice(&mut bytes, &self.b2);

        fs::write(path, bytes)
            .map_err(|err| format!("unable to write weights file {}: {err}", path.display()))
    }

    pub fn default_seeded() -> Self {
        let mut w1 = vec![0.0_f32; INPUT_DIM * HIDDEN_DIM];
        let mut b1 = vec![0.0_f32; HIDDEN_DIM];
        let mut w2 = vec![0.0_f32; STATE_DIM * HIDDEN_DIM];
        let mut b2 = vec![0.0_f32; STATE_DIM];

        for i in 0..INPUT_DIM {
            w1[i * INPUT_DIM + i] = 1.12;
        }

        for h in INPUT_DIM..HIDDEN_DIM {
            for i in 0..INPUT_DIM {
                w1[h * INPUT_DIM + i] = pseudo_signed(h as u32, i as u32) * 0.32;
            }
            b1[h] = pseudo_signed(h as u32, 911) * 0.06;
        }

        for o in 0..STATE_DIM {
            let row = o * HIDDEN_DIM;
            w2[row + o] = -0.65;
            w2[row + STATE_DIM + o] = 0.82;
            w2[row + (STATE_DIM * 2 + (o % ANCHOR_DIM))] = 0.58;

            w2[row + ((o + 1) % STATE_DIM)] += 0.08;
            w2[row + (STATE_DIM + ((o + 2) % STATE_DIM))] += 0.05;

            for h in INPUT_DIM..HIDDEN_DIM {
                w2[row + h] += pseudo_signed((o as u32) + 1337, h as u32) * 0.08;
            }

            b2[o] = -0.02;
        }

        Self { w1, b1, w2, b2 }
    }

    fn validate(&self) -> Result<(), String> {
        let expected_w1 = INPUT_DIM * HIDDEN_DIM;
        let expected_b1 = HIDDEN_DIM;
        let expected_w2 = STATE_DIM * HIDDEN_DIM;
        let expected_b2 = STATE_DIM;

        if self.w1.len() != expected_w1 {
            return Err(format!(
                "invalid w1 size: got {}, expected {}",
                self.w1.len(),
                expected_w1
            ));
        }
        if self.b1.len() != expected_b1 {
            return Err(format!(
                "invalid b1 size: got {}, expected {}",
                self.b1.len(),
                expected_b1
            ));
        }
        if self.w2.len() != expected_w2 {
            return Err(format!(
                "invalid w2 size: got {}, expected {}",
                self.w2.len(),
                expected_w2
            ));
        }
        if self.b2.len() != expected_b2 {
            return Err(format!(
                "invalid b2 size: got {}, expected {}",
                self.b2.len(),
                expected_b2
            ));
        }

        Ok(())
    }
}

#[allow(dead_code)]
fn append_f32_slice(target: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        target.extend_from_slice(&value.to_le_bytes());
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    if offset + 4 > bytes.len() {
        return Err("unexpected end of file while reading u32".to_owned());
    }
    let mut data = [0_u8; 4];
    data.copy_from_slice(&bytes[offset..offset + 4]);
    Ok(u32::from_le_bytes(data))
}

fn read_f32_vec(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    let byte_count = count * 4;
    if *cursor + byte_count > bytes.len() {
        return Err("unexpected end of file while reading f32 payload".to_owned());
    }

    let mut values = Vec::with_capacity(count);
    for chunk in bytes[*cursor..*cursor + byte_count].chunks_exact(4) {
        let mut data = [0_u8; 4];
        data.copy_from_slice(chunk);
        values.push(f32::from_le_bytes(data));
    }

    *cursor += byte_count;
    Ok(values)
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn pseudo_signed(a: u32, b: u32) -> f32 {
    let h = hash_u32(a.wrapping_mul(0x9e37_79b9) ^ b.wrapping_mul(0x7f4a_7c15));
    let n = (h & 0x000f_ffff) as f32 / 1_048_575.0;
    n * 2.0 - 1.0
}
