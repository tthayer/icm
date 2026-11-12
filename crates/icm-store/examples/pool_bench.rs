//! Evaluates the performance impact of adding a connection pool to the
//! SQLite backend (issue #1).
//!
//! The audit claim is that a single `rusqlite::Connection` behind an
//! `Arc<Mutex<..>>` — how `icm serve --http` and the web dashboard share
//! the store — serializes every read, leaving WAL's reader concurrency
//! unused. This example measures the same workload under two
//! architectures on the same database:
//!
//! * `serialized` — one `SqliteStore` (one connection) behind a `Mutex`.
//!   This is how the servers currently share the store.
//! * `pooled` — one independent read-only connection per worker thread,
//!   the shape a connection pool would hand out.
//!
//! Two workloads are run so the two effects don't get confused:
//!
//! * `point`  — cheap primary-key reads. Isolates the serialization /
//!   head-of-line-blocking effect of the shared connection.
//! * `search` — lexical (`search_fts`) reads, the realistic recall path.
//!
//! For each workload it reports a single-thread reference and the scaling
//! factor (throughput at N threads ÷ throughput at 1 thread). A shared
//! connection cannot exceed 1.0x; a pool can.
//!
//! Run with:
//! `cargo run -p icm-store --example pool_bench --release`
//!
//! Tunable via env: `BENCH_SEED`, `BENCH_THREADS`, `BENCH_POINT_REQS`,
//! `BENCH_SEARCH_REQS`, `BENCH_ROUNDS`.

use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use icm_core::{Importance, Memory, MemoryStore};
use icm_store::SqliteStore;

const DIMS: usize = 64;
/// Primary-key reads per point request.
const GETS_PER_REQUEST: usize = 20;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy, PartialEq)]
enum Workload {
    Point,
    Search,
}

fn point_request(s: &SqliteStore, ids: &[String], cursor: &mut usize) {
    for _ in 0..GETS_PER_REQUEST {
        let id = &ids[*cursor % ids.len()];
        *cursor = cursor.wrapping_add(1);
        let _ = s.get(id).unwrap();
    }
}

fn search_request(s: &SqliteStore) {
    for term in ["alpha", "beta", "gamma", "delta"] {
        let _ = s.search_fts(term, 50).unwrap();
    }
}

struct Ctx {
    ids: Vec<String>,
}

fn do_request(s: &SqliteStore, w: Workload, ctx: &Ctx, cursor: &mut usize) {
    match w {
        Workload::Point => point_request(s, &ctx.ids, cursor),
        Workload::Search => search_request(s),
    }
}

#[derive(Debug)]
struct Stats {
    total: Duration,
    requests: usize,
    p50: Duration,
    p99: Duration,
}

impl Stats {
    fn new(total: Duration, mut lat: Vec<Duration>, requests: usize) -> Self {
        lat.sort_unstable();
        let pick = |q: f64| -> Duration {
            if lat.is_empty() {
                return Duration::ZERO;
            }
            let idx = (((lat.len() - 1) as f64) * q).round() as usize;
            lat[idx.min(lat.len() - 1)]
        };
        Stats {
            total,
            requests,
            p50: pick(0.50),
            p99: pick(0.99),
        }
    }

    fn throughput(&self) -> f64 {
        self.requests as f64 / self.total.as_secs_f64()
    }

    fn report(&self, label: &str) {
        println!(
            "  {label:<34} tput={:>10.0} req/s  p50={:>8.2}ms  p99={:>8.2}ms",
            self.throughput(),
            self.p50.as_secs_f64() * 1e3,
            self.p99.as_secs_f64() * 1e3,
        );
    }
}

fn run_serialized(
    path: &std::path::Path,
    w: Workload,
    ctx: &Arc<Ctx>,
    threads: usize,
    reqs: usize,
) -> Stats {
    let store = Arc::new(Mutex::new(SqliteStore::open_readonly(path).unwrap()));
    let barrier = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let ctx = Arc::clone(ctx);
            thread::spawn(move || {
                let mut lat = Vec::with_capacity(reqs);
                let mut cursor = t * 7919;
                barrier.wait();
                for _ in 0..reqs {
                    let now = Instant::now();
                    {
                        // Exactly the sharing pattern of the HTTP/web servers.
                        let guard = store.lock().unwrap();
                        do_request(&guard, w, &ctx, &mut cursor);
                    }
                    lat.push(now.elapsed());
                }
                lat
            })
        })
        .collect();
    barrier.wait();
    let start = Instant::now();
    let mut all = Vec::with_capacity(threads * reqs);
    for h in handles {
        all.extend(h.join().unwrap());
    }
    let total = start.elapsed();
    Stats::new(total, all, threads * reqs)
}

fn run_pooled(
    path: &std::path::Path,
    w: Workload,
    ctx: &Arc<Ctx>,
    threads: usize,
    reqs: usize,
) -> Stats {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let path = path.to_path_buf();
            let barrier = Arc::clone(&barrier);
            let ctx = Arc::clone(ctx);
            thread::spawn(move || {
                // One connection per worker: what a pool checkout would give.
                let store = SqliteStore::open_readonly(&path).unwrap();
                let mut cursor = t * 7919;
                // Warm before the timed region.
                do_request(&store, w, &ctx, &mut cursor);
                let mut lat = Vec::with_capacity(reqs);
                barrier.wait();
                for _ in 0..reqs {
                    let now = Instant::now();
                    do_request(&store, w, &ctx, &mut cursor);
                    lat.push(now.elapsed());
                }
                lat
            })
        })
        .collect();
    barrier.wait();
    let start = Instant::now();
    let mut all = Vec::with_capacity(threads * reqs);
    for h in handles {
        all.extend(h.join().unwrap());
    }
    let total = start.elapsed();
    Stats::new(total, all, threads * reqs)
}

fn bench_workload(
    path: &std::path::Path,
    ctx: &Arc<Ctx>,
    w: Workload,
    name: &str,
    threads: usize,
    reqs: usize,
    rounds: usize,
) {
    // Best-of-N: the box is shared, so take the fastest (least-interfered) run.
    let mut solo = f64::MAX;
    let mut ser = 0.0_f64;
    let mut pool = 0.0_f64;
    let mut ser_stats = None;
    let mut pool_stats = None;
    for _ in 0..rounds {
        let s = run_pooled(path, w, ctx, 1, reqs);
        solo = solo.min(s.throughput());
        let s = run_serialized(path, w, ctx, threads, reqs);
        if s.throughput() > ser {
            ser = s.throughput();
            ser_stats = Some(s);
        }
        let s = run_pooled(path, w, ctx, threads, reqs);
        if s.throughput() > pool {
            pool = s.throughput();
            pool_stats = Some(s);
        }
    }
    println!(
        "\nworkload `{name}`: {threads} threads x {reqs} requests (best of {rounds})"
    );
    println!("  {:<34} tput={:>10.0} req/s  (single-thread reference)", "", solo);
    if let Some(st) = ser_stats {
        st.report(&format!("serialized (1 conn, {threads} thr)"));
    }
    if let Some(st) = pool_stats {
        st.report(&format!("pooled ({threads} conns, {threads} thr)"));
    }
    println!(
        "  scaling vs 1 thread:  serialized {:.2}x   pooled {:.2}x",
        ser / solo,
        pool / solo,
    );
}

fn main() {
    let seed_rows = env_usize("BENCH_SEED", 5_000);
    let threads = env_usize("BENCH_THREADS", 8);
    let point_reqs = env_usize("BENCH_POINT_REQS", 4_000);
    let search_reqs = env_usize("BENCH_SEARCH_REQS", 50);
    let rounds = env_usize("BENCH_ROUNDS", 3);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.db");

    print!("seeding {seed_rows} memories ... ");
    let t = Instant::now();
    let mut ids = Vec::with_capacity(seed_rows);
    {
        let store = SqliteStore::with_dims(&path, DIMS).unwrap();
        for i in 0..seed_rows {
            let mut m = Memory::new(
                "bench".into(),
                format!("alpha beta gamma delta entry {i} with some filler words"),
                Importance::Medium,
            );
            m.keywords = vec!["alpha".into(), "bench".into()];
            let id = store.store(m).unwrap();
            ids.push(id);
        }
    }
    println!("done in {:.2}s", t.elapsed().as_secs_f64());

    let ctx = Arc::new(Ctx { ids });

    bench_workload(
        &path,
        &ctx,
        Workload::Point,
        "point",
        threads,
        point_reqs,
        rounds,
    );
    bench_workload(
        &path,
        &ctx,
        Workload::Search,
        "search",
        threads,
        search_reqs,
        rounds,
    );
}
