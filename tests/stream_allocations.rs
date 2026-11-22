use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use fst::{Set, Streamer};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static WORDS: &str = include_str!("../data/words-10000");

fn count_allocs_during_traversal(set: &Set<Vec<u8>>) -> (usize, usize) {
    ALLOCS.store(0, Ordering::SeqCst);
    let mut stream = set.stream();
    let mut n = 0;
    while let Some(key) = stream.next() {
        std::hint::black_box(key);
        n += 1;
    }
    (n, ALLOCS.load(Ordering::SeqCst))
}

/// Iterating a stream must reuse its internal key buffer. In particular,
/// the number of heap allocations performed while traversing a stream must
/// be bounded by a small constant (buffer growth only), and must not scale
/// with the number of keys in the stream.
#[test]
fn stream_traversal_does_not_allocate_per_key() {
    let set = Set::from_iter(WORDS.lines()).unwrap();

    // Warm up once so that lazy one-time allocations (if any) happen before
    // measurement.
    count_allocs_during_traversal(&set);

    let (n, allocs) = count_allocs_during_traversal(&set);
    assert_eq!(n, WORDS.lines().count());
    // A stream that allocates a fresh buffer for every key would perform at
    // least `n` allocations. Buffer reuse means we only see a handful of
    // allocations for the stream's internal buffers, regardless of `n`.
    assert!(
        allocs < 100,
        "traversing {} keys caused {} heap allocations; \
         expected buffer reuse (a small constant number of allocations)",
        n,
        allocs
    );
}
