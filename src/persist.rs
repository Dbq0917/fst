/*!
Durable, atomic construction of FST files.

The streaming [`Builder`](crate::raw::Builder) keeps its working memory
bounded (the registry and the unfinished-node stack are both fixed size),
so an FST of arbitrary size can be built without growing the heap in
proportion to the amount of data inserted. The data is instead streamed to
an [`io::Write`](std::io::Write). When that writer is a normal file opened
with [`File::create`](std::fs::File::create), the destination path is
overwritten *in place*: a reader that opens the path while the build is in
progress observes a truncated or partially written file and silently gets
back an incomplete index.

[`AtomicFile`] solves the replacement half of the problem. It is a
write-only file handle that writes to a temporary file in the same
directory as the destination and, once the entire FST has been written and
fsynced, atomically renames the temporary file over the destination. A
rename is atomic on POSIX file systems, so a reader either opens the old,
complete file or the new, complete file, but never a half-written one.

Combining the two gives a construction path whose heap usage stays
constant while the on-disk index grows, and whose publish step is atomic
with respect to concurrent readers:

```
use fst::SetBuilder;

let keys = ["fa", "fo", "fob", "foo", "food"];
let mut builder = SetBuilder::create("/tmp/index.fst")?;
for key in &keys {
    builder.insert(key)?;
}
builder.persist()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
*/

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::Result;

/// A process-wide counter used to make temporary file names unique even when
/// several builds start within the same nanosecond.
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// A write-only file that is atomically swapped into place on success.
///
/// An `AtomicFile` writes to a temporary file located in the same directory
/// as the intended destination. The destination itself is not touched until
/// [`persist`](AtomicFile::persist) is called. That method flushes and
/// fsyncs the temporary file and then atomically renames it over the
/// destination, so any reader opening the destination path observes either
/// the previous complete file or the newly written complete file, never a
/// partially written one.
///
/// If the handle is dropped without calling `persist` (for example because
/// building the FST returned an error midway through), the temporary file is
/// removed and the destination is left untouched.
///
/// The temporary file lives in the same directory as the destination because
/// [`rename`](std::fs::rename) is only guaranteed to be atomic when the
/// source and destination reside on the same file system.
pub struct AtomicFile {
    /// The destination path that the temporary file is renamed onto.
    dest: PathBuf,
    /// The path of the temporary file currently being written.
    tmp: PathBuf,
    /// The open temporary file. Wrapped in an `Option` so that `persist` can
    /// consume and sync it without implementing `Drop` ordering tricks.
    file: Option<File>,
    /// Set to true once the temporary file has been published, which keeps
    /// `Drop` from deleting the renamed file.
    persisted: bool,
}

impl AtomicFile {
    /// Create a new atomic writer whose contents will eventually be published
    /// at `dest`.
    ///
    /// A fresh temporary file is created immediately in the same directory
    /// as `dest`. The destination is not modified in any way until
    /// [`persist`](AtomicFile::persist) succeeds.
    pub fn create<P: AsRef<Path>>(dest: P) -> Result<AtomicFile> {
        let dest = dest.as_ref().to_path_buf();
        let dir = parent_dir(&dest);
        let tmp = unique_tmp_path(&dir, &dest);
        let file =
            OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        Ok(AtomicFile { dest, tmp, file: Some(file), persisted: false })
    }

    /// Flush, fsync and atomically publish the written data at the
    /// destination path.
    ///
    /// The temporary file is synced to disk before the rename, and the parent
    /// directory is synced afterwards on Unix so that the rename itself
    /// survives a crash. On success the destination points at a complete,
    /// durable file. The temporary path is consumed and is not left behind.
    pub fn persist(mut self) -> Result<()> {
        let mut file = self.file.take().expect("AtomicFile already finished");
        file.flush()?;
        file.sync_all()?;
        fs::rename(&self.tmp, &self.dest)?;
        sync_parent_dir(&self.dest)?;
        self.persisted = true;
        Ok(())
    }

    /// Return the destination path that [`persist`](AtomicFile::persist) will
    /// write to.
    pub fn destination(&self) -> &Path {
        &self.dest
    }

    /// Return the path of the temporary file that receives the in-progress
    /// writes.
    pub fn temp_path(&self) -> &Path {
        &self.tmp
    }
}

impl io::Write for AtomicFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.as_mut().expect("AtomicFile already finished").write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.file.as_mut() {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        // Drop the open file handle first so the unlink succeeds on every
        // platform, then remove the temporary file unless it was renamed
        // into place.
        drop(self.file.take());
        if !self.persisted {
            let _ = fs::remove_file(&self.tmp);
        }
    }
}

/// Return the directory that temporary files for `dest` should be created
/// in. A destination without a parent component (e.g. `"index.fst"`) is
/// treated as relative to the current directory.
fn parent_dir(dest: &Path) -> PathBuf {
    match dest.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// Build a temporary path in `dir` that does not exist yet.
///
/// The name is derived from the destination file name plus the process id,
/// a process-wide monotonically increasing counter and the current time in
/// nanoseconds. In the astronomically unlikely event of a collision, a
/// handful of extra counter-based attempts are made before giving up.
fn unique_tmp_path(dir: &Path, dest: &Path) -> PathBuf {
    let stem = dest
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "fst".to_string());
    let pid = std::process::id();
    for _ in 0..16 {
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp_name = format!(".{}.{}.{}.{}.tmp", stem, pid, seq, nanos);
        let tmp = dir.join(tmp_name);
        if !tmp.exists() {
            return tmp;
        }
    }
    // `AtomicFile::create` opens with `create_new`, which still fails
    // safely if the path somehow exists, so returning a fresh candidate is
    // acceptable here.
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".{}.{}.{}.tmp", stem, pid, seq))
}

/// fsync the directory containing `path` so that a preceding rename is
/// durable. Directory fsync is a Unix concept; on other platforms this is a
/// no-op (the rename itself remains atomic).
fn sync_parent_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let dir = parent_dir(path);
        let handle = File::open(dir)?;
        // `sync_all` on a directory descriptor issues fsync(2) on Linux,
        // which makes the preceding rename durable across a crash.
        handle.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::AtomicFile;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("fst-atomic-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn publish_replaces_destination_atomically() {
        let dest = tmp_path("publish.fst");
        std::fs::write(&dest, b"old contents").unwrap();

        let mut af = AtomicFile::create(&dest).unwrap();
        // While building, the destination must remain the old file and the
        // temporary file must hold the partial contents.
        af.write_all(b"new ").unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"old contents");
        assert!(af.temp_path().exists());
        af.write_all(b"contents").unwrap();
        af.persist().unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"new contents");
        let _ = std::fs::remove_file(&dest);
    }

    #[test]
    fn drop_without_persist_keeps_destination() {
        let dest = tmp_path("drop.fst");
        std::fs::write(&dest, b"old contents").unwrap();
        let tmp = {
            let mut af = AtomicFile::create(&dest).unwrap();
            af.write_all(b"partial").unwrap();
            let tmp = af.temp_path().to_path_buf();
            assert!(tmp.exists());
            tmp
        };
        assert_eq!(std::fs::read(&dest).unwrap(), b"old contents");
        assert!(!tmp.exists());
        let _ = std::fs::remove_file(&dest);
    }

    #[test]
    fn concurrent_builders_use_distinct_temp_files() {
        let dest = tmp_path("concurrent.fst");
        let a = AtomicFile::create(&dest).unwrap();
        let b = AtomicFile::create(&dest).unwrap();
        assert_ne!(a.temp_path(), b.temp_path());
        let tmp_a = a.temp_path().to_path_buf();
        let tmp_b = b.temp_path().to_path_buf();
        drop((a, b));
        assert!(!tmp_a.exists());
        assert!(!tmp_b.exists());
        assert!(!dest.exists());
    }
}
