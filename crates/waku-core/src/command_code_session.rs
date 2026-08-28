//! Command Code native transcript helpers.
//!
//! Sessions are append-only JSONL files under `~/.commandcode/projects/<slug>/`.
//! The first line is a `{type:"session", version:3, id, timestamp, cwd}` header;
//! every later line is a tree entry with `id` / `parentId`. User turns are
//! `{type:"message", message:{role:"user"}}`. Rewind and branch copy a retained
//! prefix into a new session file so the original transcript stays untouched.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, anyhow, bail};
use chrono::{SecondsFormat, Utc};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::model::ProviderResumeCursor;

pub fn session_title(session_id: &str) -> anyhow::Result<Option<String>> {
    let path = find_session_file(session_id)?;
    if let Some(title) = title_from_meta(&path) {
        return Ok(Some(title));
    }
    Ok(title_from_transcript(&read_records(&path)?))
}

pub fn fork_dropping_turns(
    session_id: &str,
    turns_to_remove: usize,
) -> anyhow::Result<ProviderResumeCursor> {
    let source = find_session_file(session_id)?;
    let records = read_records(&source)?;
    let (header, entries) = split_header(records)?;
    let retained = retain_entries(&entries, turns_to_remove)?;
    write_fork(&source, header, retained)
}

fn projects_directory() -> anyhow::Result<PathBuf> {
    dirs::home_dir()
        .map(|home| home.join(".commandcode").join("projects"))
        .ok_or_else(|| anyhow!("Command Code's configuration directory could not be located"))
}

fn find_session_file(session_id: &str) -> anyhow::Result<PathBuf> {
    let filename = format!("{session_id}.jsonl");
    let projects = projects_directory()?;
    if !projects.is_dir() {
        bail!(
            "Command Code session directory {} does not exist",
            projects.display()
        );
    }
    for entry in fs::read_dir(&projects).with_context(|| {
        format!(
            "could not read Command Code's session directory at {}",
            projects.display()
        )
    })? {
        let Ok(entry) = entry else {
            continue;
        };
        let candidate = entry.path().join(&filename);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!("Command Code session {session_id} was not found on disk")
}

fn read_records(path: &Path) -> anyhow::Result<Vec<Value>> {
    let file = fs::File::open(path)
        .with_context(|| format!("could not open Command Code session {}", path.display()))?;
    Ok(BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(&line).ok())
        .collect())
}

fn split_header(mut records: Vec<Value>) -> anyhow::Result<(Value, Vec<Value>)> {
    if records
        .first()
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str)
        != Some("session")
    {
        bail!("Command Code session is missing its header");
    }
    let header = records.remove(0);
    Ok((header, records))
}

fn is_user_message(entry: &Value) -> bool {
    entry.get("type").and_then(Value::as_str) == Some("message")
        && entry.pointer("/message/role").and_then(Value::as_str) == Some("user")
}

fn active_branch(entries: &[Value]) -> Vec<&Value> {
    if entries.is_empty() {
        return Vec::new();
    }
    let by_id: HashMap<&str, &Value> = entries
        .iter()
        .filter_map(|entry| {
            entry
                .get("id")
                .and_then(Value::as_str)
                .map(|id| (id, entry))
        })
        .collect();
    let mut current = entries
        .last()
        .and_then(|entry| entry.get("id").and_then(Value::as_str));
    let mut chain = Vec::new();
    let mut seen = 0usize;
    while let Some(id) = current {
        let Some(entry) = by_id.get(id) else {
            break;
        };
        chain.push(*entry);
        seen += 1;
        if seen > entries.len() {
            break;
        }
        current = entry
            .get("parentId")
            .and_then(Value::as_str)
            .filter(|parent| !parent.is_empty());
    }
    chain.reverse();
    if chain.is_empty() {
        entries.iter().collect()
    } else {
        chain
    }
}

fn retain_entries(entries: &[Value], turns_to_remove: usize) -> anyhow::Result<Vec<Value>> {
    let branch: Vec<Value> = active_branch(entries).into_iter().cloned().collect();
    let user_count = branch.iter().filter(|entry| is_user_message(entry)).count();
    if turns_to_remove > user_count {
        bail!("cannot drop {turns_to_remove} Command Code turns from a session with {user_count}");
    }
    let retain = user_count.saturating_sub(turns_to_remove);
    if retain == 0 {
        return Ok(Vec::new());
    }
    let mut kept = Vec::new();
    let mut users = 0usize;
    for entry in branch {
        if is_user_message(&entry) {
            if users == retain {
                break;
            }
            users += 1;
        }
        kept.push(entry);
    }
    Ok(kept)
}

fn write_fork(
    source: &Path,
    mut header: Value,
    entries: Vec<Value>,
) -> anyhow::Result<ProviderResumeCursor> {
    let directory = source
        .parent()
        .ok_or_else(|| anyhow!("Command Code session has no project directory"))?;
    let session_id = Uuid::new_v4().to_string();
    let original_id = header.get("id").and_then(Value::as_str).map(str::to_owned);
    header["id"] = json!(session_id);
    header["timestamp"] = json!(Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true));
    if let Some(ref original_id) = original_id {
        header["parentSession"] = json!(original_id);
    }

    let path = directory.join(format!("{session_id}.jsonl"));
    write_jsonl(&path, &header, &entries)?;
    write_meta(directory, &session_id, original_id.as_deref(), &entries);
    Ok(ProviderResumeCursor::CommandCode { session_id })
}

fn write_jsonl(path: &Path, header: &Value, entries: &[Value]) -> anyhow::Result<()> {
    let mut options = OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options
        .open(path)
        .with_context(|| format!("could not write Command Code session {}", path.display()))?;
    writeln!(file, "{}", serde_json::to_string(header)?)?;
    for entry in entries {
        writeln!(file, "{}", serde_json::to_string(entry)?)?;
    }
    Ok(())
}

fn write_meta(directory: &Path, session_id: &str, parent: Option<&str>, entries: &[Value]) {
    let mut meta = json!({ "entrypoint": "print" });
    if let Some(parent) = parent {
        meta["parentSessionId"] = json!(parent);
    }
    if let Some(title) = title_from_transcript_entries(entries) {
        meta["title"] = json!(title);
    }
    let _ = fs::write(
        directory.join(format!("{session_id}.meta.json")),
        serde_json::to_vec_pretty(&meta).unwrap_or_default(),
    );
}

fn title_from_meta(session_path: &Path) -> Option<String> {
    let meta_path = session_path.with_extension("meta.json");
    let value: Value = serde_json::from_slice(&fs::read(meta_path).ok()?).ok()?;
    value
        .get("title")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(str::to_owned)
}

fn title_from_transcript(records: &[Value]) -> Option<String> {
    let entries = if records
        .first()
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str)
        == Some("session")
    {
        &records[1..]
    } else {
        records
    };
    title_from_transcript_entries(entries)
}

fn title_from_transcript_entries(entries: &[Value]) -> Option<String> {
    entries.iter().rev().find_map(|entry| {
        (entry.get("type").and_then(Value::as_str) == Some("session_info"))
            .then(|| {
                entry
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(str::to_owned)
            })
            .flatten()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(id: &str, parent: Option<&str>) -> Value {
        json!({
            "type": "message",
            "id": id,
            "parentId": parent,
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": { "role": "user", "content": id }
        })
    }

    fn assistant(id: &str, parent: &str) -> Value {
        json!({
            "type": "message",
            "id": id,
            "parentId": parent,
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": { "role": "assistant", "content": "ok" }
        })
    }

    #[test]
    fn retain_keeps_the_prefix_of_user_turns() {
        let entries = vec![
            user("u1", None),
            assistant("a1", "u1"),
            user("u2", Some("a1")),
            assistant("a2", "u2"),
            user("u3", Some("a2")),
            assistant("a3", "u3"),
        ];
        let kept = retain_entries(&entries, 1).unwrap();
        assert_eq!(
            kept.iter()
                .filter_map(|entry| entry.get("id").and_then(Value::as_str))
                .collect::<Vec<_>>(),
            ["u1", "a1", "u2", "a2"]
        );
        assert!(retain_entries(&entries, 3).unwrap().is_empty());
    }

    #[test]
    fn title_reads_the_latest_session_info_entry() {
        let entries = vec![
            json!({"type":"session_info","id":"t1","parentId":null,"timestamp":"t","name":"First"}),
            user("u1", Some("t1")),
            json!({"type":"session_info","id":"t2","parentId":"u1","timestamp":"t","name":"Better title"}),
        ];
        assert_eq!(
            title_from_transcript_entries(&entries).as_deref(),
            Some("Better title")
        );
    }
}
