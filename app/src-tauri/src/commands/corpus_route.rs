use serde::Serialize;
use serde_json::{Map, Value};

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CorpusRouteBin {
    pub id: i64,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
}

pub(super) fn tour_of(obj: &Map<String, Value>) -> Vec<i64> {
    let Some(items) = array_of(obj, &["tour"]) else {
        return Vec::new();
    };
    items.iter().filter_map(tour_stop).collect()
}

pub(super) fn tour_indices_of(obj: &Map<String, Value>) -> Vec<i64> {
    let Some(items) = array_of(obj, &["tour_indices", "tourIndices"]) else {
        return Vec::new();
    };
    items.iter().filter_map(as_i64).collect()
}

pub(super) fn route_bins_of(obj: &Map<String, Value>) -> Vec<CorpusRouteBin> {
    if let Some(items) = obj.get("all_bin_coords").and_then(Value::as_array) {
        let parsed: Vec<CorpusRouteBin> = items.iter().filter_map(coord_bin).collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    let Some(items) = obj.get("bins").and_then(Value::as_array) else {
        return Vec::new();
    };
    items.iter().filter_map(paired_bin).collect()
}

fn array_of<'a>(obj: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Vec<Value>> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(Value::as_array))
}

fn tour_stop(value: &Value) -> Option<i64> {
    if let Some(id) = as_i64(value) {
        return Some(id);
    }
    let obj = value.as_object()?;
    ["id", "bin_id", "binId"]
        .iter()
        .find_map(|key| obj.get(*key).and_then(as_i64))
}

fn coord_bin(value: &Value) -> Option<CorpusRouteBin> {
    let bin = named_bin(value)?;
    if bin.lat.is_none() && bin.lon.is_none() {
        None
    } else {
        Some(bin)
    }
}

fn paired_bin(value: &Value) -> Option<CorpusRouteBin> {
    let bin = named_bin(value)?;
    if bin.lat.is_some() && bin.lon.is_some() {
        Some(bin)
    } else {
        None
    }
}

fn named_bin(value: &Value) -> Option<CorpusRouteBin> {
    let obj = value.as_object()?;
    let id = ["id", "bin_id", "binId"]
        .iter()
        .find_map(|key| obj.get(*key).and_then(as_i64))?;
    Some(CorpusRouteBin {
        id,
        lat: super::first_f64(obj, &["lat", "latitude"]),
        lon: super::first_f64(obj, &["lng", "lon", "longitude"]),
    })
}

pub(super) fn mean_of_array(items: &[Value]) -> (Option<f64>, Option<f64>) {
    let mut sum_lat = 0.0;
    let mut sum_lon = 0.0;
    let mut count = 0.0;
    for item in items {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(lat) = super::first_f64(obj, &["lat", "latitude"]) else {
            continue;
        };
        let Some(lon) = super::first_f64(obj, &["lng", "lon", "longitude"]) else {
            continue;
        };
        sum_lat += lat;
        sum_lon += lon;
        count += 1.0;
    }
    if count == 0.0 {
        (None, None)
    } else {
        (Some(sum_lat / count), Some(sum_lon / count))
    }
}

fn as_i64(value: &Value) -> Option<i64> {
    let number = super::as_number(value)?;
    number.as_i64().or_else(|| integral_float(&number))
}

fn integral_float(number: &serde_json::Number) -> Option<i64> {
    let parsed = number.as_f64()?;
    if !parsed.is_finite() || parsed.fract() != 0.0 {
        return None;
    }
    let text = parsed.to_string();
    let digits = text.strip_suffix(".0").unwrap_or(&text);
    digits.parse().ok()
}
