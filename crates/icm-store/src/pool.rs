//! A pool of independent [`Store`] connections over one database file (#1).
//!
//! A `rusqlite::Connection` is `Send` but not `Sync`, so a single connection
//! shared behind a `Mutex` serializes every request: that is exactly what the
//! HTTP API and web dashboard did (`Arc<Mutex<Store>>`), and it made WAL's
//! reader concurrency unreachable. Measured on a 12-core host, throughput was
//! flat (~1.0x) at 2, 4 and 8 threads through one connection; a pool lets the
//! reads run in parallel.
//!
//! Members are created up front. [`StorePool::checkout`] hands one out and
//! **blocks** (it does not fail) while all are busy, so a server can never drop
//! a request for want of a connection. The guard is deliberately `!Send`, like
//! `MutexGuard`, so it cannot be held across an `await`.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::{Condvar, Mutex};

use icm_core::IcmResult;

use crate::backend::Store;

/// A pool of [`Store`] connections over the same database.
pub struct StorePool {
    idle: Mutex<Vec<Store>>,
    available: Condvar,
    /// Members created at construction; constant (the pool does not grow).
    size: usize,
}

fn open_member(path: &Path, dims: usize, readonly: bool) -> IcmResult<Store> {
    if readonly {
        Store::open_readonly(path)
    } else {
        Store::with_dims(path, dims)
    }
}

impl StorePool {
    /// Open `size` members over `path`. `readonly` opens every member
    /// read-only (the `--read-only serve` mode).
    pub fn new(path: &Path, dims: usize, readonly: bool, size: usize) -> IcmResult<Self> {
        let size = size.max(1);
        let mut members = Vec::with_capacity(size);
        for _ in 0..size {
            members.push(open_member(path, dims, readonly)?);
        }
        Ok(Self {
            idle: Mutex::new(members),
            available: Condvar::new(),
            size,
        })
    }

    /// Build a pool around an already-open `seed` store, adding `size - 1`
    /// more members over `path`. Lets a caller reuse the warm store it just
    /// opened (migrations, backup, read-only probe already done on it).
    pub fn with_seed(
        seed: Store,
        path: &Path,
        dims: usize,
        readonly: bool,
        size: usize,
    ) -> IcmResult<Self> {
        let size = size.max(1);
        let mut members = Vec::with_capacity(size);
        members.push(seed);
        for _ in 1..size {
            members.push(open_member(path, dims, readonly)?);
        }
        Ok(Self {
            idle: Mutex::new(members),
            available: Condvar::new(),
            size,
        })
    }

    /// Wrap a single existing store (in-memory or already-open). The pool
    /// cannot grow, so concurrent checkouts beyond one will block — intended
    /// for tests and one-shot callers.
    pub fn from_single(store: Store) -> Self {
        Self {
            idle: Mutex::new(vec![store]),
            available: Condvar::new(),
            size: 1,
        }
    }

    /// Members the pool holds in total.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Members currently idle.
    #[must_use]
    pub fn idle_len(&self) -> usize {
        self.idle.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Check out a member, blocking while all are busy. Never fails.
    pub fn checkout(&self) -> PooledStore<'_> {
        let mut idle = self.idle.lock().unwrap_or_else(|e| e.into_inner());
        while idle.is_empty() {
            idle = self.available.wait(idle).unwrap_or_else(|e| e.into_inner());
        }
        let store = idle.pop().expect("checked non-empty above");
        PooledStore {
            pool: self,
            store: Some(store),
            _not_send: PhantomData,
        }
    }
}

/// RAII handle to one [`Store`] from a [`StorePool`]. Derefs to `Store` and
/// returns the member to the pool on drop.
pub struct PooledStore<'a> {
    pool: &'a StorePool,
    store: Option<Store>,
    /// `!Send`/`!Sync`, mirroring `MutexGuard`: a guard held across an `await`
    /// could otherwise block a runtime worker in [`StorePool::checkout`].
    _not_send: PhantomData<*const ()>,
}

impl Deref for PooledStore<'_> {
    type Target = Store;
    fn deref(&self) -> &Store {
        self.store.as_ref().expect("PooledStore::store taken only in drop")
    }
}

impl DerefMut for PooledStore<'_> {
    fn deref_mut(&mut self) -> &mut Store {
        self.store.as_mut().expect("PooledStore::store taken only in drop")
    }
}

impl Drop for PooledStore<'_> {
    fn drop(&mut self) {
        if let Some(store) = self.store.take() {
            let mut idle = self.pool.idle.lock().unwrap_or_else(|e| e.into_inner());
            idle.push(store);
            self.pool.available.notify_one();
        }
    }
}

/// Default number of connections for a serving process: the host's available
/// parallelism, capped, overridable with `ICM_SQLITE_POOL`.
#[must_use]
pub fn default_pool_size() -> usize {
    if let Ok(v) = std::env::var("ICM_SQLITE_POOL")
        && let Ok(n) = v.trim().parse::<usize>()
        && n > 0
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4)
        .clamp(2, 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use icm_core::{Importance, Memory, MemoryStore};

    fn mem(topic: &str, summary: &str) -> Memory {
        Memory::new(topic.into(), summary.into(), Importance::Medium)
    }

    #[test]
    fn from_single_returns_the_member_to_the_pool() {
        let pool = StorePool::from_single(Store::in_memory().unwrap());
        assert_eq!(pool.size(), 1);
        {
            let s = pool.checkout();
            assert!(s.count().is_ok());
            assert_eq!(pool.idle_len(), 0, "checked out");
        }
        assert_eq!(pool.idle_len(), 1, "returned on drop");
        // Usable again.
        assert!(pool.checkout().count().is_ok());
    }

    #[test]
    fn file_pool_serves_members_concurrently() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pool.db");
        {
            let s = Store::with_dims(&path, 64).unwrap();
            s.store(mem("t", "alpha")).unwrap();
        }

        let pool = std::sync::Arc::new(StorePool::new(&path, 64, false, 4).unwrap());
        assert_eq!(pool.size(), 4);

        // Four threads each check out and read; all should get their own
        // connection without blocking each other.
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = std::sync::Arc::clone(&pool);
                std::thread::spawn(move || {
                    let s = pool.checkout();
                    s.count().unwrap()
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), 1);
        }
        assert_eq!(pool.idle_len(), 4, "all members returned");
    }

    #[test]
    fn with_seed_reuses_the_seed_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seed.db");
        let seed = Store::with_dims(&path, 64).unwrap();
        seed.store(mem("t", "beta")).unwrap();
        let pool = StorePool::with_seed(seed, &path, 64, false, 3).unwrap();
        assert_eq!(pool.size(), 3);
        for _ in 0..3 {
            assert_eq!(pool.checkout().count().unwrap(), 1);
        }
        assert_eq!(pool.idle_len(), 3);
    }

    #[test]
    fn default_pool_size_is_bounded() {
        let n = default_pool_size();
        assert!((1..=8).contains(&n), "got {n}");
    }
}
