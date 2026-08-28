//! Command Code's print-mode session.
//!
//! `command-code --print --output-format json` is one-shot: the process reads
//! the prompt from stdin, emits NDJSON event frames plus a final `result`
//! line, and exits. There is no ACP, no stream-json input, and no mid-turn
//! steer. Waku therefore keeps a driver for the conversation and spawns a
//! fresh child on every prompt, passing `--resume <id>` once the native
//! session exists. Exiting the child is not `ProcessExited` — that would drop
//! the runtime and lose the next prompt's resume id.
//!
//! Flags and payloads were read off the real CLI (v1.36). Headless `standard`
//! permission mode blocks writes rather than asking, so Supervised degrades
//! to that; Plan, auto-accept, and yolo map onto launch flags.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

use anyhow::{Context as _, anyhow};
use crossbeam_channel::{Sender, unbounded};
use parking_lot::Mutex;
use serde_json::Value;

use super::activity;
use crate::driver::{
    DriverControl, DriverEventSender, DriverEventSink, DriverStartOptions, SessionOptions,
};
use crate::model::{ActivityKind, DriverEvent, InteractionMode, ProviderResumeCursor, RuntimeMode};

enum CommandMessage {
    Prompt(String),
    Shutdown,
}

struct LiveOptions {
    mode: RuntimeMode,
    interaction_mode: InteractionMode,
    model: Option<String>,
    reasoning_effort: Option<String>,
}

pub struct CommandCodeDriver {
    commands: Sender<CommandMessage>,
    active_pid: Arc<AtomicU32>,
    session_id: Arc<Mutex<Option<String>>>,
    options: Arc<Mutex<LiveOptions>>,
}

/// Print-mode arguments. The prompt is written to stdin, never as an argument.
pub(super) fn print_args(
    session_id: Option<&str>,
    model: Option<&str>,
    reasoning_effort: Option<&str>,
    mode: RuntimeMode,
    interaction_mode: InteractionMode,
) -> Vec<String> {
    let mut args = vec![
        "--print".to_owned(),
        "--output-format".to_owned(),
        "json".to_owned(),
        "--skip-onboarding".to_owned(),
        "--trust".to_owned(),
        "--no-auto-update".to_owned(),
        "--tools-enable".to_owned(),
        "todo_write".to_owned(),
    ];
    args.extend(permission_args(mode, interaction_mode));
    if let Some(session_id) = session_id.filter(|id| !id.is_empty()) {
        args.extend(["--resume".to_owned(), session_id.to_owned()]);
    }
    if let Some(model) = model.filter(|model| !model.is_empty()) {
        args.extend(["--model".to_owned(), model.to_owned()]);
    }
    if let Some(effort) = reasoning_effort.filter(|effort| !effort.is_empty()) {
        args.extend(["--effort".to_owned(), effort.to_owned()]);
    }
    args
}

fn permission_args(mode: RuntimeMode, interaction_mode: InteractionMode) -> Vec<String> {
    match (interaction_mode, mode) {
        (InteractionMode::Plan, _) | (_, RuntimeMode::Plan) => vec!["--plan".to_owned()],
        (_, RuntimeMode::Ask) => vec!["--permission-mode".to_owned(), "standard".to_owned()],
        (_, RuntimeMode::AutoAcceptEdits | RuntimeMode::Auto) => vec!["--auto-accept".to_owned()],
        (_, RuntimeMode::FullAccess) => vec!["--yolo".to_owned()],
    }
}

impl CommandCodeDriver {
    pub fn start(options: DriverStartOptions, events: DriverEventSender) -> anyhow::Result<Self> {
        let DriverStartOptions {
            binary,
            cwd,
            mode,
            interaction_mode,
            model,
            reasoning_effort,
            service_tier: _,
            context_window: _,
            agent_preset: _,
            computer_use_enabled: _,
            provider_cursor,
        } = options;
        let session_id = match provider_cursor {
            Some(ProviderResumeCursor::CommandCode { session_id }) => {
                (!session_id.is_empty()).then_some(session_id)
            }
            Some(cursor) => {
                return Err(anyhow!(
                    "cannot resume Command Code from a {} cursor",
                    cursor.provider().display_name()
                ));
            }
            None => None,
        };

        if let Some(session_id) = session_id.clone() {
            let _ = events.send(DriverEvent::Connected {
                provider_cursor: Some(ProviderResumeCursor::CommandCode {
                    session_id: session_id.clone(),
                }),
            });
        }

        let (commands, command_rx) = unbounded();
        let active_pid = Arc::new(AtomicU32::new(0));
        let session_id = Arc::new(Mutex::new(session_id));
        let live = Arc::new(Mutex::new(LiveOptions {
            mode,
            interaction_mode,
            model,
            reasoning_effort,
        }));

        let writer_events = events;
        let writer_pid = active_pid.clone();
        let writer_session = session_id.clone();
        let writer_options = live.clone();
        thread::Builder::new()
            .name("waku-commandcode-writer".into())
            .spawn(move || {
                while let Ok(message) = command_rx.recv() {
                    match message {
                        CommandMessage::Prompt(text) => run_prompt(
                            &binary,
                            &cwd,
                            &text,
                            &writer_options,
                            &writer_session,
                            &writer_pid,
                            &writer_events,
                        ),
                        CommandMessage::Shutdown => break,
                    }
                }
            })?;

        Ok(Self {
            commands,
            active_pid,
            session_id,
            options: live,
        })
    }
}

impl DriverControl for CommandCodeDriver {
    fn prompt(&self, prompt: String) {
        let _ = self.commands.send(CommandMessage::Prompt(prompt));
    }

    fn cancel(&self) {
        let pid = self.active_pid.load(Ordering::Relaxed);
        if pid != 0 {
            #[cfg(unix)]
            {
                let _ = Command::new("/bin/kill")
                    .args(["-INT", &pid.to_string()])
                    .status();
            }
        }
    }

    fn respond(&self, _request_id: String, _option_id: String) {}

    fn apply_options(&self, options: SessionOptions) -> bool {
        let mut live = self.options.lock();
        live.mode = options.mode;
        live.interaction_mode = options.interaction_mode;
        live.model = options.model;
        live.reasoning_effort = options.reasoning_effort;
        true
    }

    fn rollback(&self, turns: usize) -> anyhow::Result<Option<ProviderResumeCursor>> {
        if turns == 0 {
            return Ok(None);
        }
        self.fork(turns).map(Some)
    }

    fn fork(&self, turns_to_remove: usize) -> anyhow::Result<ProviderResumeCursor> {
        let session_id = self
            .session_id
            .lock()
            .clone()
            .ok_or_else(|| anyhow!("Command Code's native session is unavailable"))?;
        crate::command_code_session::fork_dropping_turns(&session_id, turns_to_remove)
    }
}

impl Drop for CommandCodeDriver {
    fn drop(&mut self) {
        let _ = self.commands.send(CommandMessage::Shutdown);
        self.cancel();
    }
}

fn run_prompt(
    binary: &std::path::Path,
    cwd: &std::path::Path,
    prompt: &str,
    options: &Mutex<LiveOptions>,
    session_id: &Mutex<Option<String>>,
    active_pid: &AtomicU32,
    events: &DriverEventSender,
) {
    let _ = events.send(DriverEvent::TurnStarted);
    let (mode, interaction_mode, model, reasoning_effort, resume) = {
        let live = options.lock();
        (
            live.mode,
            live.interaction_mode,
            live.model.clone(),
            live.reasoning_effort.clone(),
            session_id.lock().clone(),
        )
    };
    let mut command = crate::command_env::command(binary);
    command.current_dir(cwd).args(print_args(
        resume.as_deref(),
        model.as_deref(),
        reasoning_effort.as_deref(),
        mode,
        interaction_mode,
    ));
    let command = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match crate::command_env::spawn(command)
        .context("failed to start `command-code` in print mode")
    {
        Ok(child) => child,
        Err(error) => {
            let _ = events.send(DriverEvent::Error(error.to_string()));
            let _ = events.send(DriverEvent::TurnFinished {
                success: false,
                summary: Some(tr!(
                    "errors.provider_receive_prompt",
                    provider = "Command Code"
                )),
            });
            return;
        }
    };

    if let Err(error) = write_prompt(&mut child, prompt) {
        let _ = events.send(DriverEvent::Error(tr!(
            "errors.provider_transport_write",
            provider = "Command Code",
            error = error
        )));
        let _ = child.kill();
        let _ = child.wait();
        let _ = events.send(DriverEvent::TurnFinished {
            success: false,
            summary: Some(tr!(
                "errors.provider_receive_prompt",
                provider = "Command Code"
            )),
        });
        return;
    }

    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = events.send(DriverEvent::Error(tr!(
                "errors.provider_transport_write_short",
                provider = "Command Code"
            )));
            let _ = child.kill();
            let _ = child.wait();
            let _ = events.send(DriverEvent::TurnFinished {
                success: false,
                summary: Some(tr!(
                    "errors.provider_stopped_receiving",
                    provider = "Command Code"
                )),
            });
            return;
        }
    };
    let stderr = child.stderr.take();
    active_pid.store(child.id(), Ordering::Relaxed);

    let last_visible_stderr = Arc::new(Mutex::new(None::<String>));
    let stderr_thread = stderr.and_then(|stderr| {
        let stderr_events = events.clone();
        let stderr_last_error = last_visible_stderr.clone();
        thread::Builder::new()
            .name("waku-commandcode-stderr".into())
            .spawn(move || {
                let lines = BufReader::new(stderr)
                    .lines()
                    .map_while(Result::ok)
                    .filter(|line| !line.trim().is_empty())
                    .collect::<Vec<_>>();
                if let Some(message) = super::support::provider_stderr_error(lines) {
                    let error = format!("Command Code: {message}");
                    *stderr_last_error.lock() = Some(error.clone());
                    let _ = stderr_events.send(DriverEvent::Error(error));
                }
            })
            .ok()
    });

    let mut state = StreamState::default();
    let title_refresh = super::title_refresh::NativeTitleRefresh::default();
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let was_connected = state.connected;
        handle_line(&value, events, session_id, &mut state);
        if state.connected && (!was_connected || state.turn_settled) && !state.titled {
            if let Some(id) = session_id.lock().clone() {
                title_refresh.start(
                    "waku-commandcode-title",
                    vec![
                        std::time::Duration::from_millis(500),
                        std::time::Duration::from_secs(2),
                    ],
                    events.clone(),
                    move || crate::command_code_session::session_title(&id),
                );
            }
        }
    }

    let status = child.wait();
    active_pid.store(0, Ordering::Relaxed);
    if let Some(thread) = stderr_thread {
        let _ = thread.join();
    }

    if !state.turn_settled {
        if let Ok(status) = status
            && !status.success()
            && last_visible_stderr.lock().is_none()
            && !state.saw_interrupt
        {
            let _ = events.send(DriverEvent::Error(tr!(
                "errors.provider_exited",
                provider = "Command Code",
                status = status
            )));
        }
        let _ = events.send(DriverEvent::TurnFinished {
            success: state.result_success.unwrap_or(false),
            summary: state.result_error.clone(),
        });
    }
}

fn write_prompt(child: &mut Child, prompt: &str) -> std::io::Result<()> {
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("Command Code stdin unavailable"))?;
    stdin.write_all(prompt.as_bytes())?;
    if !prompt.ends_with('\n') {
        stdin.write_all(b"\n")?;
    }
    stdin.flush()
}

#[derive(Default)]
struct StreamState {
    tools: HashMap<String, (ActivityKind, String)>,
    saw_text: bool,
    saw_reasoning: bool,
    saw_interrupt: bool,
    connected: bool,
    titled: bool,
    turn_settled: bool,
    result_success: Option<bool>,
    result_error: Option<String>,
}

fn handle_line(
    value: &Value,
    events: &impl DriverEventSink,
    session_id: &Mutex<Option<String>>,
    state: &mut StreamState,
) {
    match value.get("type").and_then(Value::as_str) {
        Some("event") => {
            if let Some(event) = value.get("event") {
                handle_event(event, events, session_id, state);
            }
        }
        Some("result") => handle_result(value, events, session_id, state),
        _ => handle_event(value, events, session_id, state),
    }
}

fn handle_event(
    event: &Value,
    events: &impl DriverEventSink,
    session_id: &Mutex<Option<String>>,
    state: &mut StreamState,
) {
    match event.get("type").and_then(Value::as_str) {
        Some("run_start") => {
            if let Some(id) = json_str(event, "sessionId") {
                remember_session(id, events, session_id, state);
            }
        }
        Some("text_delta") => {
            if let Some(text) = json_str(event, "delta").filter(|text| !text.is_empty()) {
                state.saw_text = true;
                let _ = events.send(DriverEvent::TextDelta(text.to_owned()));
            }
        }
        Some("thinking_delta") => {
            if let Some(text) = json_str(event, "delta").filter(|text| !text.is_empty()) {
                state.saw_reasoning = true;
                let _ = events.send(DriverEvent::ReasoningDelta(text.to_owned()));
            }
        }
        Some("thinking_end") if !state.saw_reasoning => {
            if let Some(text) = json_str(event, "text").filter(|text| !text.is_empty()) {
                state.saw_reasoning = true;
                let _ = events.send(DriverEvent::ReasoningDelta(text.to_owned()));
            }
        }
        Some("tool_queued" | "tool_running") => {
            emit_tool(event, events, state, false, false, None);
        }
        Some("tool_update") => {
            emit_tool(event, events, state, false, false, event.get("partial"));
        }
        Some("tool_completed") => {
            emit_tool(event, events, state, false, true, event.get("result"));
        }
        Some("tool_errored") => {
            emit_tool(
                event,
                events,
                state,
                true,
                true,
                event.get("error").or_else(|| event.get("result")),
            );
        }
        Some("session_titled") => {
            if let Some(title) = json_str(event, "title")
                .map(str::trim)
                .filter(|title| !title.is_empty())
            {
                state.titled = true;
                let _ = events.send(DriverEvent::AutoTitleUpdated(Some(title.to_owned())));
            }
        }
        Some("model_request_end") => {
            if let Some(usage) = event.get("usage")
                && let Some(tokens) = usage_tokens(usage)
            {
                let _ = events.send(DriverEvent::UsageUpdated {
                    context_tokens: Some(tokens),
                    context_window: None,
                });
            }
        }
        Some("interrupted") => {
            state.saw_interrupt = true;
        }
        Some("run_error") => {
            if let Some(message) = error_text(event.get("error")) {
                let _ = events.send(DriverEvent::Error(format!("Command Code: {message}")));
                state.result_success = Some(false);
                state.result_error = Some(message);
            }
        }
        _ => {}
    }
}

fn handle_result(
    value: &Value,
    events: &impl DriverEventSink,
    session_id: &Mutex<Option<String>>,
    state: &mut StreamState,
) {
    if let Some(id) = json_str(value, "sessionId") {
        remember_session(id, events, session_id, state);
    }
    if let Some(usage) = value.get("usage")
        && let Some(tokens) = usage_tokens(usage)
    {
        let _ = events.send(DriverEvent::UsageUpdated {
            context_tokens: Some(tokens),
            context_window: None,
        });
    }
    if !state.saw_text
        && let Some(text) = json_str(value, "finalText").filter(|text| !text.is_empty())
    {
        state.saw_text = true;
        let _ = events.send(DriverEvent::TextDelta(text.to_owned()));
    }
    let subtype = json_str(value, "subtype").unwrap_or("success");
    let success = subtype == "success" && !state.saw_interrupt;
    if !success && let Some(message) = error_text(value.get("error")) {
        let _ = events.send(DriverEvent::Error(format!("Command Code: {message}")));
        state.result_error = Some(message);
    }
    state.result_success = Some(success);
    state.turn_settled = true;
    let _ = events.send(DriverEvent::TurnFinished {
        success,
        summary: state.result_error.clone(),
    });
}

fn remember_session(
    id: &str,
    events: &impl DriverEventSink,
    session_id: &Mutex<Option<String>>,
    state: &mut StreamState,
) {
    let mut stored = session_id.lock();
    let changed = stored.as_deref() != Some(id);
    if changed {
        *stored = Some(id.to_owned());
    }
    drop(stored);
    if !state.connected || changed {
        state.connected = true;
        let _ = events.send(DriverEvent::Connected {
            provider_cursor: Some(ProviderResumeCursor::CommandCode {
                session_id: id.to_owned(),
            }),
        });
    }
}

fn emit_tool(
    event: &Value,
    events: &impl DriverEventSink,
    state: &mut StreamState,
    failed: bool,
    complete: bool,
    output: Option<&Value>,
) {
    let id = json_str(event, "toolCallId").map(str::to_owned);
    let wire_name = json_str(event, "toolName").unwrap_or("tool");
    let kind = super::support::classify_tool(wire_name);
    let title = activity::input_title(event.get("input"))
        .or_else(|| {
            json_str(event, "description")
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        })
        .or_else(|| {
            id.as_deref()
                .and_then(|id| state.tools.get(id).map(|(_, title)| title.clone()))
        })
        .unwrap_or_else(|| wire_name.to_owned());
    if let Some(id) = id.clone() {
        state.tools.insert(id, (kind, title.clone()));
    }
    let _ = events.send(DriverEvent::RichActivity(activity::tool_activity(
        id,
        kind,
        title,
        event.get("input"),
        output,
        None,
        failed,
        complete,
    )));
}

fn json_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn error_text(value: Option<&Value>) -> Option<String> {
    let value = value?;
    json_str(value, "message")
        .or_else(|| json_str(value, "error"))
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            value
                .as_str()
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        })
}

fn usage_tokens(usage: &Value) -> Option<u64> {
    let field = |name: &str| {
        usage
            .get(name)
            .and_then(Value::as_u64)
            .or_else(|| usage.get(name).and_then(Value::as_f64).map(|n| n as u64))
            .unwrap_or(0)
    };
    let total = field("inputTokens")
        + field("outputTokens")
        + field("cacheReadTokens")
        + field("cacheWriteTokens")
        + field("input_tokens")
        + field("output_tokens");
    (total > 0).then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::test_event_channel;
    use crossbeam_channel::unbounded;
    use serde_json::json;

    #[test]
    fn print_args_resume_model_effort_and_permission_flags() {
        let args = print_args(
            Some("sess-1"),
            Some("deepseek/deepseek-v4-flash"),
            Some("high"),
            RuntimeMode::FullAccess,
            InteractionMode::Build,
        );
        assert!(args.contains(&"--print".to_owned()));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--output-format", "json"])
        );
        assert!(args.contains(&"--skip-onboarding".to_owned()));
        assert!(args.contains(&"--trust".to_owned()));
        assert!(args.contains(&"--yolo".to_owned()));
        assert!(args.windows(2).any(|pair| pair == ["--resume", "sess-1"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--model", "deepseek/deepseek-v4-flash"])
        );
        assert!(args.windows(2).any(|pair| pair == ["--effort", "high"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--tools-enable", "todo_write"])
        );
        assert!(!args.iter().any(|arg| arg.contains("hello")));

        let plan = print_args(None, None, None, RuntimeMode::Ask, InteractionMode::Plan);
        assert!(plan.contains(&"--plan".to_owned()));
        assert!(!plan.contains(&"--resume".to_owned()));

        let ask = print_args(None, None, None, RuntimeMode::Ask, InteractionMode::Build);
        assert!(
            ask.windows(2)
                .any(|pair| pair == ["--permission-mode", "standard"])
        );

        let auto = print_args(
            None,
            None,
            None,
            RuntimeMode::AutoAcceptEdits,
            InteractionMode::Build,
        );
        assert!(auto.contains(&"--auto-accept".to_owned()));
    }

    #[test]
    fn streams_events_then_settles_on_the_result_line() {
        let (events, event_rx) = unbounded();
        let session_id = Mutex::new(None);
        let mut state = StreamState::default();
        let wire = [
            json_event("run_start", json!({"sessionId":"abc-123"})),
            json_event("thinking_delta", json!({"delta":"pondering"})),
            json_event(
                "tool_queued",
                json!({"toolCallId":"t1","toolName":"Bash","input":{"command":"ls"}}),
            ),
            json_event(
                "tool_completed",
                json!({"toolCallId":"t1","toolName":"Bash","result":"a.txt"}),
            ),
            json_event("text_delta", json!({"delta":"BANANA."})),
            json_event("session_titled", json!({"title":"Banana check"})),
            json!({
                "type": "result",
                "subtype": "success",
                "sessionId": "abc-123",
                "finalText": "ignored",
                "usage": { "inputTokens": 10, "outputTokens": 4 }
            }),
        ];
        for message in wire {
            handle_line(&message, &events, &session_id, &mut state);
        }

        let mut seen = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            seen.push(event);
        }
        assert!(matches!(
            &seen[0],
            DriverEvent::Connected {
                provider_cursor: Some(ProviderResumeCursor::CommandCode { session_id })
            } if session_id == "abc-123"
        ));
        assert!(matches!(&seen[1], DriverEvent::ReasoningDelta(t) if t == "pondering"));
        assert!(matches!(&seen[2], DriverEvent::RichActivity(item)
                if item.kind == ActivityKind::Command && !item.complete));
        assert!(matches!(&seen[3], DriverEvent::RichActivity(item)
                if item.complete && item.output.as_deref() == Some("a.txt")));
        assert!(matches!(&seen[4], DriverEvent::TextDelta(t) if t == "BANANA."));
        assert!(matches!(
            &seen[5],
            DriverEvent::AutoTitleUpdated(Some(title)) if title == "Banana check"
        ));
        assert!(matches!(
            &seen[6],
            DriverEvent::UsageUpdated {
                context_tokens: Some(14),
                ..
            }
        ));
        assert!(matches!(
            &seen[7],
            DriverEvent::TurnFinished { success: true, .. }
        ));
        assert_eq!(session_id.lock().as_deref(), Some("abc-123"));
        assert!(state.turn_settled);
        assert!(state.saw_text);
    }

    #[test]
    fn result_final_text_fills_in_when_deltas_never_arrived() {
        let (events, event_rx) = unbounded();
        let session_id = Mutex::new(None);
        let mut state = StreamState::default();
        handle_line(
            &json!({
                "type": "result",
                "subtype": "success",
                "sessionId": "s1",
                "finalText": "hello from the result line"
            }),
            &events,
            &session_id,
            &mut state,
        );
        let mut seen = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            seen.push(event);
        }
        assert!(seen.iter().any(
            |event| matches!(event, DriverEvent::TextDelta(t) if t == "hello from the result line")
        ));
        assert!(
            seen.iter()
                .any(|event| matches!(event, DriverEvent::TurnFinished { success: true, .. }))
        );
    }

    #[test]
    fn apply_options_is_absorbed_for_the_next_spawn() {
        let (events, _rx) = test_event_channel();
        let driver = CommandCodeDriver::start(
            DriverStartOptions {
                binary: std::path::PathBuf::from("/bin/true"),
                cwd: std::env::temp_dir(),
                mode: RuntimeMode::FullAccess,
                interaction_mode: InteractionMode::Build,
                model: None,
                reasoning_effort: None,
                service_tier: None,
                context_window: None,
                agent_preset: None,
                computer_use_enabled: false,
                provider_cursor: None,
            },
            events,
        )
        .unwrap();
        assert!(driver.apply_options(SessionOptions {
            mode: RuntimeMode::Ask,
            interaction_mode: InteractionMode::Plan,
            model: Some("moonshotai/kimi-k2.5".into()),
            reasoning_effort: Some("low".into()),
            service_tier: None,
            context_window: None,
        }));
        let live = driver.options.lock();
        assert_eq!(live.mode, RuntimeMode::Ask);
        assert_eq!(live.interaction_mode, InteractionMode::Plan);
        assert_eq!(live.model.as_deref(), Some("moonshotai/kimi-k2.5"));
        assert_eq!(live.reasoning_effort.as_deref(), Some("low"));
    }

    /// Drives the real CLI through the actual driver, including a second turn
    /// that `--resume`s the session the first print created. Ignored by
    /// default: needs the CLI installed, credentials, and the network.
    #[test]
    #[ignore = "requires an installed, authenticated command-code"]
    fn command_code_print_session_against_the_real_cli() {
        let binary = crate::command_env::find_executable("command-code")
            .expect("command-code is not installed");
        let (events, event_rx) = crate::driver::test_event_channel();
        let driver = CommandCodeDriver::start(
            DriverStartOptions {
                binary,
                cwd: std::env::temp_dir(),
                mode: RuntimeMode::FullAccess,
                interaction_mode: InteractionMode::Build,
                model: None,
                reasoning_effort: None,
                service_tier: None,
                context_window: None,
                agent_preset: None,
                computer_use_enabled: false,
                provider_cursor: None,
            },
            events,
        )
        .expect("the print session should start");

        let collect = |driver: &CommandCodeDriver, prompt: &str| -> String {
            driver.prompt(prompt.to_owned());
            let mut text = String::new();
            while let Ok(event) = event_rx.recv_timeout(std::time::Duration::from_secs(180)) {
                match event {
                    DriverEvent::TextDelta(delta) => text.push_str(&delta),
                    DriverEvent::TurnFinished { success, .. } => {
                        assert!(success, "the turn should settle successfully");
                        return text;
                    }
                    DriverEvent::Error(error) => panic!("the CLI reported: {error}"),
                    _ => {}
                }
            }
            panic!("the turn never settled");
        };

        let first = collect(&driver, "Reply with exactly: BANANA. Use no tools.");
        assert!(first.contains("BANANA"), "expected a reply, got {first:?}");

        let second = collect(
            &driver,
            "What word did I just ask you to reply with? Answer with that word only.",
        );
        assert!(
            second.contains("BANANA"),
            "the resumed session should retain context across turns, got {second:?}"
        );
    }

    fn json_event(event_type: &str, mut payload: Value) -> Value {
        payload
            .as_object_mut()
            .expect("event payload")
            .insert("type".into(), Value::String(event_type.to_owned()));
        json!({"type": "event", "event": payload})
    }
}
