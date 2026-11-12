//! Single-user latency profile: serial store / recall / open operations, the
//! shape of an agent workflow that touches the store a few times a minute.
//!
//! Unlike `pool_bench`, there is no concurrency here: it measures the latency
//! of one operation at a time, which is what a single user feels.
//!
//! Run with:
//! `cargo run -p icm-store --example single_user_bench --release`
//!
//! Compare write durability settings with:
//!   `ICM_SQLITE_SYNCHRONOUS=FULL cargo run ... --release`
//! Tunables: `BENCH_SEED`, `BENCH_WRITES`, `BENCH_READS`.

use std::time::{Duration, Instant};

use icm_core::{
    Concept, ConceptLink, Importance, Memoir, MemoirStore, Memory, MemoryStore, Relation,
};
use icm_store::SqliteStore;

const DIMS: usize = 64;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn pct(mut v: Vec<Duration>, q: f64) -> Duration {
    v.sort_unstable();
    if v.is_empty() {
        return Duration::ZERO;
    }
    let i = (((v.len() - 1) as f64) * q).round() as usize;
    v[i.min(v.len() - 1)]
}

fn report(label: &str, lats: &[Duration]) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mean = lats.iter().map(|d| d.as_secs_f64()).sum::<f64>() / lats.len() as f64 * 1e3;
    println!(
        "  {label:<22} n={:<5} mean={:>7.2}ms  p50={:>7.2}ms  p90={:>7.2}ms  p99={:>7.2}ms",
        lats.len(),
        mean,
        ms(pct(lats.to_vec(), 0.50)),
        ms(pct(lats.to_vec(), 0.90)),
        ms(pct(lats.to_vec(), 0.99)),
    );
}

fn emb(i: usize) -> Vec<f32> {    (0..DIMS)
        .map(|d| ((i + d) % 13) as f32 / 13.0)
        .collect()
}

/// Memoir operations, the other half of the single-user workflow: cycle-checked
/// link creation and graph neighborhood expansion both do one query per node
/// (issue #8), so their cost grows with memoir size.
fn memoir_profile(k: usize) {
    let store = SqliteStore::in_memory().unwrap();
    let mid = store
        .create_memoir(Memoir::new("bench".into(), "profile".into()))
        .unwrap();
    let mut cids = Vec::with_capacity(k);
    for i in 0..k {
        let c = Concept::new(mid.clone(), format!("c{i}"), format!("concept {i}"));
        cids.push(store.add_concept(c).unwrap());
    }

    // Forward chain c0 -> c1 -> ... -> c{k-1}. Each add_link runs the cycle
    // BFS, but against a short outgoing path, so this is cheap.
    let t = Instant::now();
    for i in 1..k {
        store
            .add_link(ConceptLink::new(
                cids[i - 1].clone(),
                cids[i].clone(),
                Relation::RelatedTo,
            ))
            .unwrap();
    }
    let build = t.elapsed();

    // One add_link that must walk the whole chain: c{k-1} -> c0 closes a cycle,
    // so the BFS visits all k nodes before rejecting.
    let t = Instant::now();
    let _ = store.add_link(ConceptLink::new(
        cids[k - 1].clone(),
        cids[0].clone(),
        Relation::RelatedTo,
    ));
    let cycle = t.elapsed();

    // get_neighborhood(depth=k): BFS over the whole graph, one query per node
    // per direction.
    let t = Instant::now();
    let _ = store.get_neighborhood(&cids[0], k).unwrap();
    let hood = t.elapsed();

    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!(
        "  memoir k={k:<4} build_chain={:>8.1}ms  cycle_check(1 link)={:>7.2}ms  get_neighborhood={:>8.1}ms",
        ms(build),
        ms(cycle),
        ms(hood),
    );
}

fn main() {
    let seed = env_usize("BENCH_SEED", 2_000);
    let writes = env_usize("BENCH_WRITES", 500);
    let reads = env_usize("BENCH_READS", 500);
    let sync = std::env::var("ICM_SQLITE_SYNCHRONOUS").unwrap_or_else(|_| "NORMAL(default)".into());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("user.db");

    let store = SqliteStore::with_dims(&path, DIMS).unwrap();
    for i in 0..seed {
        let mut m = Memory::new(
            "seeded".into(),
            format!("seeded alpha beta entry {i} with filler words"),
            Importance::Medium,
        );
        m.embedding = Some(emb(i));
        store.store(m).unwrap();
    }

    let db_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    // --- write latency: one store() per op, the hook / `icm store` path ---
    let mut wl = Vec::with_capacity(writes);
    for i in 0..writes {
        let mut m = Memory::new(
            "live".into(),
            format!("live alpha beta entry {i} something worth remembering"),
            Importance::Medium,
        );
        m.embedding = Some(emb(i + 1));
        let t = Instant::now();
        store.store(m).unwrap();
        wl.push(t.elapsed());
    }

    // --- recall latency: lexical + vector, one recall per op ---
    let q = emb(0);
    let mut rl = Vec::with_capacity(reads);
    for _ in 0..reads {
        let t = Instant::now();
        let _ = store.search_fts("alpha beta", 5).unwrap();
        let _ = store.search_by_embedding(&q, 5).unwrap();
        rl.push(t.elapsed());
    }

    // --- open latency: schema_is_current runs on every process start ---
    let mut ol = Vec::with_capacity(20);
    for _ in 0..20 {
        let t = Instant::now();
        let s = SqliteStore::with_dims(&path, DIMS).unwrap();
        ol.push(t.elapsed());
        drop(s);
    }

    println!(
        "single-user profile: seed={seed} rows, db={:.1} MiB, synchronous={sync}\n",
        db_bytes as f64 / (1024.0 * 1024.0)
    );
    report("store()  (write)", &wl);
    report("recall    (read)", &rl);
    report("open      (process)", &ol);

    println!("\nmemoir operations (issue #8 — one query per node):");
    for k in [50usize, 200, 500] {
        memoir_profile(k);
    }
}
