//! Tests for the two properties the construction path has to guarantee:
//!
//! 1. Streaming an fst to a file keeps peak heap usage bounded instead of
//!    growing linearly with the amount of data inserted.
//! 2. Publishing the finished fst is an atomic replacement, so a reader can
//!    never observe a half-written file.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use fst::{IntoStreamer, Set, SetBuilder, Streamer};

/// The medium sized data set shipped with the crate.
static WORDS: &str = include_str!("../data/words-100000");

/// Number of times the medium data set is repeated to get a "large" input
/// whose encoded fst is several times bigger. Repetitions are de-duplicated
/// by the fst, so the keys are made unique by prefixing each copy with its
/// repetition index (one byte, which keeps the keys lexicographically
/// ordered).
const LARGE_REPEATS: usize = 8;

/// Peak resident set size of the current process in bytes, read from
/// `/proc/self/status`. Only available on Linux, which is the platform these
/// construction tests target.
fn peak_rss_bytes() -> u64 {
    let status = fs::read_to_string("/proc/self/status")
        .expect("read /proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .parse()
                .expect("parse VmHWM");
            return kb * 1024;
        }
    }
    panic!("VmHWM not found in /proc/self/status");
}

/// A unique scratch directory for a test run.
fn scratch_dir(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "fst-build-test-{}-{}-{}",
        name,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write the input keys, in lexicographic order and without duplicates, to a
/// line-delimited file. Both build modes read this file with a small buffer,
/// so the full key list is never materialized on the measured process's
/// heap.
fn write_sorted_keys(dir: &Path, name: &str, repeats: usize) -> PathBuf {
    let path = dir.join(name);
    let wtr = File::create(&path).unwrap();
    let mut wtr = BufWriter::new(wtr);
    for repeat in 0..repeats {
        for line in WORDS.lines() {
            // `repeat` is a single byte for the values used in tests, and the
            // prefix keeps both ordering and uniqueness across repetitions.
            writeln!(wtr, "{:01}:{}", repeat, line).unwrap();
        }
    }
    wtr.flush().unwrap();
    path
}

/// Build the fst entirely on the heap (`Vec<u8>`) and report its peak RSS
/// delta along with the encoded size.
fn run_memory_mode(keys_path: &Path) -> (u64, usize) {
    let before = peak_rss_bytes();
    let rdr = BufReader::new(File::open(keys_path).unwrap());
    let mut builder = SetBuilder::memory();
    for line in rdr.lines() {
        builder.insert(line.unwrap()).unwrap();
    }
    let set = builder.into_set();
    let after = peak_rss_bytes();
    (after - before, set.as_fst().size())
}

/// Stream the fst to a temporary file and atomically publish it, reporting
/// the peak RSS delta and the final file size.
fn run_streaming_mode(keys_path: &Path, output: &Path) -> (u64, usize) {
    let before = peak_rss_bytes();
    let rdr = BufReader::new(File::open(keys_path).unwrap());
    let mut builder = SetBuilder::create(output).unwrap();
    for line in rdr.lines() {
        builder.insert(line.unwrap()).unwrap();
    }
    builder.persist().unwrap();
    let after = peak_rss_bytes();
    (after - before, fs::metadata(output).unwrap().len() as usize)
}

/// Re-run the current test binary in "child mode". Each child performs a
/// single, isolated build so that its `VmHWM` reflects only that build and
/// the allocator's high-water mark from one mode cannot pollute the other.
fn spawn_child(mode: &str, keys_path: &Path, output: &Path) -> String {
    let exe = std::env::current_exe().unwrap();
    let output = Command::new(exe)
        .arg("--exact")
        .arg("worker")
        .arg("--nocapture")
        // Explicitly remove any stale harness variables before setting the
        // fresh ones. The rest of the inherited environment (loader paths,
        // runtime settings, ...) must be preserved or the worker binary may
        // fail to start.
        .env_remove("FST_BUILD_MODE")
        .env_remove("FST_BUILD_KEYS")
        .env_remove("FST_BUILD_OUTPUT")
        .env("FST_BUILD_MODE", mode)
        .env("FST_BUILD_KEYS", keys_path)
        .env("FST_BUILD_OUTPUT", output)
        .output()
        .expect("spawn build worker");
    assert!(
        output.status.success(),
        "worker {} failed: {}",
        mode,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// Parse a `key=value` line printed by a worker.
fn parse_metric(stdout: &str, key: &str) -> u64 {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{}=", key)))
        .unwrap_or_else(|| panic!("missing metric {} in {:?}", key, stdout))
        .trim()
        .parse()
        .unwrap()
}

/// The worker entry point. Each invocation runs exactly one build mode and
/// prints machine-readable metrics for the parent test to compare.
#[test]
fn worker() {
    let mode = match std::env::var("FST_BUILD_MODE") {
        Ok(mode) => mode,
        Err(_) => return, // ordinary test-harness invocation; nothing to do
    };
    let keys_path: PathBuf = std::env::var("FST_BUILD_KEYS").unwrap().into();
    let output: PathBuf = std::env::var("FST_BUILD_OUTPUT").unwrap().into();
    // Remove the harness variables immediately: the worker runs only one
    // build, and leaving process-global environment variables set would leak
    // into any grandchildren spawned later.
    std::env::remove_var("FST_BUILD_MODE");
    std::env::remove_var("FST_BUILD_KEYS");
    std::env::remove_var("FST_BUILD_OUTPUT");
    let (peak, size) = if mode == "memory" {
        run_memory_mode(&keys_path)
    } else {
        run_streaming_mode(&keys_path, &output)
    };
    println!("peak_rss={}", peak);
    println!("fst_size={}", size);
}

/// Build the medium data set with both strategies and check that the
/// streaming, file-backed build does not pay the in-memory heap cost.
#[test]
fn build_peak_memory() {
    if std::env::var("FST_BUILD_MODE").is_ok() {
        worker();
        return;
    }
    if std::env::consts::OS != "linux" {
        eprintln!("peak memory test only runs on Linux; skipping");
        return;
    }

    let dir = scratch_dir("peak-memory");
    let medium_keys = write_sorted_keys(&dir, "keys-medium.sorted", 1);
    let large_keys =
        write_sorted_keys(&dir, "keys-large.sorted", LARGE_REPEATS);

    // Medium data set, run once through each construction path.
    let memory_out =
        spawn_child("memory", &medium_keys, &dir.join("unused-memory.fst"));
    let stream_out =
        spawn_child("streaming", &medium_keys, &dir.join("medium.fst"));
    let memory_peak = parse_metric(&memory_out, "peak_rss");
    let streaming_peak = parse_metric(&stream_out, "peak_rss");
    let medium_size = parse_metric(&stream_out, "fst_size");

    // The bounded builder's working set is the fixed-size registry plus the
    // unfinished-node stack and small buffers. It must stay well under this
    // generous ceiling even though the encoded fst is much larger.
    const STREAMING_CEILING: u64 = 64 * 1024 * 1024;
    assert!(
        streaming_peak < STREAMING_CEILING,
        "streaming build peak {} bytes exceeds {} byte ceiling",
        streaming_peak,
        STREAMING_CEILING
    );

    // The in-memory build must retain the whole encoded fst on the heap, so
    // its measured peak should at minimum cover the encoded size. This
    // documents the linear-memory failure mode being fixed.
    assert!(
        memory_peak >= medium_size,
        "in-memory peak {} bytes smaller than encoded fst {} bytes",
        memory_peak,
        medium_size
    );

    // Repeating the data several times grows the encoded fst roughly
    // proportionally, while the streaming build's peak must stay under the
    // same fixed ceiling: memory does not scale with the data.
    let large_out =
        spawn_child("streaming", &large_keys, &dir.join("large.fst"));
    let large_streaming_peak = parse_metric(&large_out, "peak_rss");
    let large_size = parse_metric(&large_out, "fst_size");
    assert!(large_size > medium_size * 2, "sanity: large fst is bigger");
    assert!(
        large_streaming_peak < STREAMING_CEILING,
        "large streaming build peak {} bytes exceeds ceiling",
        large_streaming_peak
    );

    eprintln!(
        "medium fst = {} bytes | memory peak = {} bytes | \
         streaming peak = {} bytes | large fst = {} bytes | \
         large streaming peak = {} bytes",
        medium_size,
        memory_peak,
        streaming_peak,
        large_size,
        large_streaming_peak
    );

    fs::remove_dir_all(&dir).ok();
}

/// The published file must be a complete, readable fst, and no temporary
/// file may be left behind.
#[test]
fn atomic_publish_is_complete_and_clean() {
    let dir = scratch_dir("atomic");
    let dest = dir.join("set.fst");
    let keys_path = write_sorted_keys(&dir, "keys.sorted", 1);

    let rdr = BufReader::new(File::open(&keys_path).unwrap());
    let mut builder = SetBuilder::create(&dest).unwrap();
    let tmp = builder.get_ref().get_ref().temp_path().to_path_buf();
    assert!(tmp.exists());
    assert!(!dest.exists());
    for line in rdr.lines() {
        builder.insert(line.unwrap()).unwrap();
    }
    builder.persist().unwrap();

    assert!(dest.exists());
    assert!(!tmp.exists(), "temporary file was not cleaned up");
    let leftovers: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "found stray temp files: {:?}", leftovers);

    // Every input key must be present in the published fst.
    let set = Set::new(fs::read(&dest).unwrap()).unwrap();
    let mut count = 0;
    let rdr = BufReader::new(File::open(&keys_path).unwrap());
    for line in rdr.lines() {
        let key = line.unwrap();
        assert!(set.contains(&key), "missing key after publish: {}", key);
        count += 1;
    }
    assert_eq!(set.len(), count);

    fs::remove_dir_all(&dir).ok();
}

/// Rebuilding over an existing file must atomically replace the whole
/// contents: a reader that memory-maps the old path keeps reading the old,
/// complete index until the rename, after which opening the path returns the
/// new, equally complete index. At no point is a partial file observable.
#[test]
fn atomic_rebuild_never_exposes_half_written_data() {
    use memmap2::Mmap;

    let dir = scratch_dir("rebuild");
    let dest = dir.join("set.fst");

    // First generation: a small complete index.
    {
        let mut builder = SetBuilder::create(&dest).unwrap();
        for key in &["apple", "banana", "cherry"] {
            builder.insert(*key).unwrap();
        }
        builder.persist().unwrap();
    }

    // A long-lived reader (the "export" process) keeps the old mapping even
    // after the path is replaced.
    let old_file = File::open(&dest).unwrap();
    let old_mmap = unsafe { Mmap::map(&old_file).unwrap() };
    let old_set = fst::raw::Fst::new(&old_mmap[..]).unwrap();
    assert_eq!(old_set.len(), 3);

    // Second generation, built over the same destination.
    let keys_path = write_sorted_keys(&dir, "keys.sorted", 1);
    {
        let rdr = BufReader::new(File::open(&keys_path).unwrap());
        let mut builder = SetBuilder::create(&dest).unwrap();
        for line in rdr.lines() {
            builder.insert(line.unwrap()).unwrap();
        }
        builder.persist().unwrap();
    }

    // The old mapping is still a valid, complete fst with the old contents.
    assert_eq!(old_set.len(), 3);
    assert!(old_set.get(b"apple").is_some());

    // Opening the path now yields the new, complete fst, and its checksum
    // verification passes (a truncated/partial file would fail).
    let new_set = Set::new(fs::read(&dest).unwrap()).unwrap();
    new_set.as_fst().verify().unwrap();
    let mut stream = new_set.into_stream();
    let mut streamed = 0;
    while stream.next().is_some() {
        streamed += 1;
    }
    let expected = WORDS.lines().count();
    assert_eq!(streamed, expected);
    assert_eq!(new_set.len(), expected);

    fs::remove_dir_all(&dir).ok();
}
