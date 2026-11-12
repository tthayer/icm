//! SQLite backend — split out of the former monolithic `store.rs`.
//!
//! `SqliteStore` and the shared row/parse helpers live in `super`
//! (`store/mod.rs`); each submodule here holds one trait impl (or a
//! coherent group of inherent methods) on that type.

use super::*;
/// Convert rusqlite::Error to IcmError::Database
pub(crate) fn db_err(e: rusqlite::Error) -> IcmError {
    IcmError::Database(e.to_string())
}

/// True when a rusqlite error is a "no such table" (a legacy DB missing an
/// FTS shadow table we optionally rebuild — issue #313).
pub(crate) fn is_missing_table(e: &rusqlite::Error) -> bool {
    e.to_string().contains("no such table")
}

/// FTS5 shadow tables maintained by ICM, checked/rebuilt during repair (#313).
pub(crate) const FTS_TABLES: [&str; 4] = [
    "memories_fts",
    "concepts_fts",
    "feedback_fts",
    "messages_fts",
];

// Shared public row types live in `crate::common` so all backends can be
// compiled into one binary without colliding definitions (issue #301).
// They are re-exported from the parent `store` module (see `mod.rs`).

/// Collect mapped rows into a Vec, converting rusqlite errors.
pub(crate) fn collect_rows<T>(
    rows: rusqlite::MappedRows<'_, impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>>,
) -> IcmResult<Vec<T>> {
    rows.collect::<Result<Vec<T>, _>>().map_err(db_err)
}

pub(crate) static SQLITE_VEC_INIT: Once = Once::new();

/// Connection-level tuning applied on every open (#6).
///
/// Chosen for a mixed read/write, WAL-mode, single-file workload:
///
/// - `cache_size=-65536` — 64 MiB page cache (SQLite's default is ~2 MiB).
/// - `mmap_size=268435456` — 256 MiB of memory-mapped reads.
/// - `temp_store=MEMORY` — sorts / `GROUP BY` / `json_each` stay in RAM.
/// - `synchronous=NORMAL` (writable only) — with WAL this drops the fsync
///   per commit. Durable across application crashes; a power loss can lose
///   the last few commits (but not corrupt the database). Set
///   `ICM_SQLITE_SYNCHRONOUS=FULL` to restore per-commit fsync.
/// - `wal_autocheckpoint=1000` (writable only) — the explicit default.
///
/// Best-effort: an unsupported pragma (e.g. `mmap_size` on an unusual VFS)
/// must not fail an open, so errors are ignored.
pub(crate) fn apply_tuning(conn: &Connection, writable: bool) {
    let synchronous = match std::env::var("ICM_SQLITE_SYNCHRONOUS") {
        Ok(v) if v.eq_ignore_ascii_case("full") => "FULL",
        _ => "NORMAL",
    };
    let mut sql = String::from(
        "PRAGMA cache_size=-65536; PRAGMA mmap_size=268435456; PRAGMA temp_store=MEMORY;",
    );
    if writable {
        sql.push_str(&format!(
            "PRAGMA synchronous={synchronous}; PRAGMA wal_autocheckpoint=1000;"
        ));
    }
    let _ = conn.execute_batch(&sql);
}


pub(crate) fn ensure_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| unsafe {
        #[allow(clippy::missing_transmute_annotations)]
        sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

/// URI-encode a filesystem path for a SQLite `file:` URI so a backslash on
/// Windows or a `?`/`#`/`%` in a pathological filename can't break the parser.
pub(crate) fn encode_sqlite_uri_path(path: &Path) -> String {
    path.to_string_lossy()
        .chars()
        .map(|c| match c {
            '?' | '#' | '%' => format!("%{:02X}", c as u32),
            // Normalize Windows backslashes; SQLite URIs accept "/".
            '\\' => "/".into(),
            other => other.to_string(),
        })
        .collect()
}

/// Open `path` strictly read-only. When `immutable` is set, add the
/// `immutable=1` URI flag — SQLite then assumes the file never changes and
/// touches no `-shm`/`-wal` sidecars, which is required on a `chmod -w`
/// parent directory (issue #263) but serves a permanently stale snapshot and
/// eventually reports spurious `SQLITE_CORRUPT` on a live DB (issue #319).
/// Plain `mode=ro` (immutable = false) is WAL-aware and sees committed
/// writes, at the cost of needing a writable directory for the sidecars.
pub(crate) fn open_readonly_uri(path: &Path, immutable: bool) -> IcmResult<Connection> {
    let encoded = encode_sqlite_uri_path(path);
    let uri = if immutable {
        format!("file:{encoded}?mode=ro&immutable=1")
    } else {
        format!("file:{encoded}?mode=ro")
    };
    let conn = Connection::open_with_flags(
        uri,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
            | rusqlite::OpenFlags::SQLITE_OPEN_URI
            | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| IcmError::Database(format!("cannot open database read-only: {e}")))?;
    apply_tuning(&conn, false);
    Ok(conn)
}

/// Open a long-lived read-only connection (issue #319).
///
/// Prefer a normal WAL-aware `mode=ro` connection: it respects locking and
/// sees writes committed after it opened — the actual deployment model for
/// `icm --read-only serve`, where hooks keep writing the same DB. Fall back to
/// `immutable=1` only when the live open can't even read the DB, e.g. a
/// `chmod -w` sandbox where SQLite can't create the `-shm` sidecar for a
/// WAL-mode file (issue #263). The read probe is essential: on such a
/// directory the open may *succeed* yet the first real read fails, so opening
/// alone is not a sufficient signal.
pub(crate) fn open_readonly_connection(path: &Path) -> IcmResult<Connection> {
    if let Ok(conn) = open_readonly_uri(path, false) {
        // Give a momentarily-locked writer time to release before deciding
        // the live open is unusable.
        let _ = conn.execute_batch("PRAGMA busy_timeout=30000;");
        // Exercise a real table read (touches the WAL/-shm path) — `SELECT 1`
        // would not.
        if conn
            .query_row("SELECT count(*) FROM sqlite_master", [], |_| Ok(()))
            .is_ok()
        {
            return Ok(conn);
        }
    }
    // Live open unusable (e.g. a read-only sandbox dir, #263). Fall back to an
    // immutable snapshot — but warn, because a long-lived reader on this
    // connection will NOT see subsequent writes (the #319 staleness tradeoff).
    tracing::warn!(
        path = %path.display(),
        "read-only DB opened immutable (sandbox fallback): writes committed after \
         this point will not be visible until the connection is reopened"
    );
    open_readonly_uri(path, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pragma_i64(conn: &Connection, name: &str) -> i64 {
        conn.query_row(&format!("PRAGMA {name};"), [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn writable_tuning_sets_cache_temp_store_and_checkpoint() {
        let conn = Connection::open_in_memory().unwrap();
        apply_tuning(&conn, true);
        assert_eq!(pragma_i64(&conn, "cache_size"), -65536);
        assert_eq!(pragma_i64(&conn, "temp_store"), 2); // MEMORY
        assert_eq!(pragma_i64(&conn, "wal_autocheckpoint"), 1000);
        // synchronous is NORMAL (1) unless ICM_SQLITE_SYNCHRONOUS=FULL.
        let sync = pragma_i64(&conn, "synchronous");
        assert!(
            sync == 1 || std::env::var("ICM_SQLITE_SYNCHRONOUS").is_ok(),
            "expected synchronous=NORMAL, got {sync}"
        );
    }

    #[test]
    fn read_only_tuning_leaves_synchronous_at_default() {
        let conn = Connection::open_in_memory().unwrap();
        apply_tuning(&conn, false);
        assert_eq!(pragma_i64(&conn, "cache_size"), -65536);
        assert_eq!(pragma_i64(&conn, "temp_store"), 2);
        // Not touched on the read-only path: SQLite's default is FULL (2).
        assert_eq!(pragma_i64(&conn, "synchronous"), 2);
    }
}
