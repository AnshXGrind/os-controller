"""
ai_pc_agent/main.py

Local Autonomous AI PC Agent — entry point.

Pipeline per command:
  mic → STT → wake-word gate → workflow match → intent interpret
  → task plan → execute steps → TTS → memory update

Usage (from workspace root):
    python -m ai_pc_agent.main

    # Diagnostics only (no mic loop):
    python -m ai_pc_agent.main --check

    # Override model:
    python -m ai_pc_agent.main --model llama3

    # Keyword-only mode (no Ollama):
    python -m ai_pc_agent.main --no-llm

    # Silent (no TTS):
    python -m ai_pc_agent.main --silent

    # Use Whisper STT:
    python -m ai_pc_agent.main --backend whisper

Press Ctrl+C to quit.

Prerequisites:
    1. ollama serve          (in a separate terminal)
    2. pip install -r ai_pc_agent/requirements.txt
"""

from __future__ import annotations
import argparse
import sys
import time

# ── Logging must be set up before any other import ────────────────────────────
from ai_pc_agent.utils.logger              import get_logger
logger = get_logger("agent.main")

# ── All other imports ─────────────────────────────────────────────────────────
from ai_pc_agent.utils                     import config
from ai_pc_agent.utils.helpers             import ollama_running, python_version, now_str

from ai_pc_agent.ai.ollama_client          import OllamaClient
from ai_pc_agent.ai.llm_reasoning          import LLMReasoning
from ai_pc_agent.ai.coding_model_client    import CodingModelClient

from ai_pc_agent.voice.speech_listener     import SpeechListener
from ai_pc_agent.voice.wake_word           import WakeWordDetector
from ai_pc_agent.voice.tts_engine          import TTSEngine

from ai_pc_agent.vision.screen_capture     import ScreenCapture
from ai_pc_agent.vision.screen_understanding import ScreenUnderstanding

from ai_pc_agent.control.system_control    import SystemControl
from ai_pc_agent.control.app_control       import AppControl
from ai_pc_agent.control.browser_control   import BrowserControl
from ai_pc_agent.control.file_control      import FileControl
from ai_pc_agent.control.keyboard_mouse    import KeyboardMouse
from ai_pc_agent.control.vscode_control    import VSCodeControl

from ai_pc_agent.memory.command_history    import CommandHistory
from ai_pc_agent.memory.context_memory     import ContextMemory
from ai_pc_agent.memory.skill_library      import SkillLibrary

from ai_pc_agent.core.intent_interpreter   import IntentInterpreter
from ai_pc_agent.core.task_planner         import TaskPlanner
from ai_pc_agent.core.command_router       import CommandRouter
from ai_pc_agent.core.self_healing_engine  import SelfHealingEngine
from ai_pc_agent.core.self_improvement_engine import SelfImprovementEngine
from ai_pc_agent.core.performance_optimizer import PerformanceOptimizer
from ai_pc_agent.core.agent_brain           import AgentBrain

from ai_pc_agent.automation.task_executor  import TaskExecutor
from ai_pc_agent.automation.workflow_engine import WorkflowEngine
from ai_pc_agent.automation.script_generator import ScriptGenerator

from ai_pc_agent.diagnostics.performance_monitor import PerformanceMonitor
from ai_pc_agent.diagnostics.error_handler       import ErrorHandler


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "ai_pc_agent",
        description = "Local Autonomous AI PC Agent (Ollama-powered)",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--check",    action="store_true",
                   help="Print diagnostics and exit")
    p.add_argument("--model",    default=None,
                   help=f"Ollama model (default: {config.get('OLLAMA_MODEL')})")
    p.add_argument("--ollama-url", default=None,
                   help="Ollama base URL (default: http://localhost:11434)")
    p.add_argument("--no-llm",   action="store_true",
                   help="Keyword-only mode — skip Ollama")
    p.add_argument("--backend",  choices=["google", "vosk", "whisper"],
                   default=config.get("STT_BACKEND", "google"))
    p.add_argument("--language", default=config.get("STT_LANGUAGE", "en-US"))
    p.add_argument("--vosk-model",    default=config.get("VOSK_MODEL_PATH"))
    p.add_argument("--whisper-model", choices=["tiny", "base", "small", "medium", "large"],
                   default=config.get("WHISPER_MODEL", "base"))
    p.add_argument("--no-wake",  action="store_true",
                   help="Disable wake-word gate (process all utterances)")
    p.add_argument("--wake-words", nargs="+", default=config.get("WAKE_WORDS"))
    p.add_argument("--silent",   action="store_true", help="Disable TTS")
    p.add_argument("--tts-rate", type=int, default=config.get("TTS_RATE", 175))
    p.add_argument("--history-file", default=config.get("HISTORY_FILE"))
    p.add_argument("--no-vision", action="store_true", help="Disable screen capture")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics banner
# ─────────────────────────────────────────────────────────────────────────────

def _print_diag(ollama: OllamaClient, monitor: PerformanceMonitor):
    health = monitor.system_health()
    models = ollama.list_models()
    print("\n" + "═" * 56)
    print("  AI PC Agent — Diagnostics")
    print("═" * 56)
    print(f"  Python      : {python_version()}")
    print(f"  Time        : {now_str()}")
    print(f"  CPU         : {health.get('cpu_pct', '?')}%")
    print(f"  RAM         : {health.get('mem_pct', '?')}%  "
          f"({health.get('mem_avail_mb', '?')} MB free)")
    print(f"  Ollama      : {ollama.base_url}")
    if models:
        print(f"  Models      : {', '.join(models)}")
        print(f"  Active model: {ollama.model}")
    else:
        print("  Ollama      : NOT REACHABLE")
        print("  → Start with: ollama serve")
    print("═" * 56 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = _build_parser().parse_args()

    # ── Diagnostics layer ─────────────────────────────────────────────────────
    monitor  = PerformanceMonitor()
    monitor.add_alert("llm", 8000)   # warn if LLM takes >8s
    monitor.add_alert("stt", 5000)

    err_handler = ErrorHandler()     # installs global sys.excepthook

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama = OllamaClient(base_url=args.ollama_url, model=args.model)

    if args.check:
        _print_diag(ollama, monitor)
        sys.exit(0)

    logger.info("═" * 56)
    logger.info("  AI PC Agent starting …")
    logger.info("═" * 56)

    llm_ok = False
    if not args.no_llm:
        with monitor.timer("ollama.check"):
            llm_ok = ollama.is_available()
        if not llm_ok:
            logger.warning(
                "Ollama not available — falling back to keyword mode. "
                "Run 'ollama serve' in another terminal."
            )

    reasoning = LLMReasoning(ollama) if llm_ok else None
    coder     = CodingModelClient(ollama)

    # ── Self-healing ──────────────────────────────────────────────────────────
    healer = SelfHealingEngine(coder)
    err_handler.add_handler(
        lambda et, ev, tb: healer.heal(str(ev), "".join(
            __import__("traceback").format_exception(et, ev, tb)
        ))
    )

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts = TTSEngine(rate=args.tts_rate, enabled=not args.silent)

    # ── Speech ────────────────────────────────────────────────────────────────
    logger.info("STT backend: %s", args.backend)
    listener = SpeechListener(
        backend         = args.backend,
        language        = args.language,
        vosk_model_path = args.vosk_model,
        whisper_model   = args.whisper_model,
    )
    with monitor.timer("stt.calibrate"):
        listener.calibrate()

    # ── Wake word ─────────────────────────────────────────────────────────────
    wake = WakeWordDetector(wake_words=args.wake_words)
    if args.no_wake:
        wake.is_triggered    = lambda _: True
        wake.strip_wake_word = lambda t: t
        logger.info("Wake-word gate DISABLED")

    # ── Vision ────────────────────────────────────────────────────────────────
    vision = None
    if not args.no_vision:
        try:
            capture = ScreenCapture()
            vision  = ScreenUnderstanding(capture, reasoning) if reasoning else None
        except Exception as exc:
            logger.warning("Vision init failed: %s", exc)

    # ── Controllers ───────────────────────────────────────────────────────────
    system  = SystemControl()
    apps    = AppControl()
    browser = BrowserControl()
    files   = FileControl()
    kb      = KeyboardMouse()
    vscode  = VSCodeControl()

    # ── Memory ────────────────────────────────────────────────────────────────
    history = CommandHistory(persist_file=args.history_file)
    context = ContextMemory()
    skills  = SkillLibrary()

    # ── Interpreter + Planner ─────────────────────────────────────────────────
    interpreter = IntentInterpreter(reasoning=reasoning, use_llm=llm_ok)
    planner     = TaskPlanner(reasoning=reasoning, interpreter=interpreter,
                              use_llm=llm_ok)

    # ── Router ────────────────────────────────────────────────────────────────
    router = CommandRouter(
        system=system, apps=apps, browser=browser,
        files=files,   kb_mouse=kb, vscode=vscode,
        history=history, skills=skills, tts=tts,
    )

    # ── Automation ────────────────────────────────────────────────────────────
    executor  = TaskExecutor(action_cooldown=float(config.get("ACTION_COOLDOWN", 0.5)))
    workflows = WorkflowEngine(executor=executor, router_fn=router.route)
    script_gen= ScriptGenerator(coder=coder)

    # ── Self-improvement ──────────────────────────────────────────────────────
    improver = SelfImprovementEngine(coder=coder, history=history, skills=skills)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = PerformanceOptimizer(monitor=monitor)

    # ── Brain ─────────────────────────────────────────────────────────────────
    brain = AgentBrain(
        router=router, interpreter=interpreter, planner=planner,
        history=history, context=context, skills=skills,
        improver=improver, vision=vision, tts=tts,
    )

    # ── Ready ─────────────────────────────────────────────────────────────────
    _print_diag(ollama, monitor)
    logger.info("Wake words: %s", wake.words)
    logger.info("All modules ready. Speak a wake word to begin.")
    logger.info("Press Ctrl+C to quit.")
    tts.speak("AI Agent online. Waiting for your command.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    sleeping    = False
    cmd_counter = 0

    try:
        while True:
            # 1. Listen
            with monitor.timer("stt.listen"):
                raw = err_handler.safe_call(
                    listener.listen_once, default=None, label="listen"
                )
            if not raw:
                continue

            logger.info("Heard: '%s'", raw)

            # 2. Wake gate
            if sleeping:
                if wake.is_triggered(raw):
                    sleeping = False
                    tts.speak("I'm awake. What can I do for you?")
                continue

            if not wake.is_triggered(raw):
                continue

            command = wake.strip_wake_word(raw)
            if not command:
                tts.speak("Yes?")
                continue

            # 3. Check built-in workflows first
            wf = workflows.find_by_trigger(command)
            if wf:
                import re as _re
                m = _re.search(r"(?:for|about|on)\s+(.+)$", command, _re.I)
                ctx = {"query": m.group(1).strip()} if m else {}
                results = workflows.run(wf, context=ctx)
                ok  = all(r.success for r in results)
                msg = f"Workflow '{wf.name}' complete." if ok else "Some steps failed."
                tts.speak(msg)
                history.add(raw_text=raw, intent=f"workflow:{wf.name}", value=command, success=ok)
                continue

            # 4. Brain → plan → execute
            with monitor.timer("brain.process"):
                success, response = err_handler.safe_call(
                    brain.process, command, default=(False, "Sorry, I couldn't process that."),
                    label="brain.process",
                )

            if response:
                tts.speak(response)

            # 5. Periodic self-improvement (every 20 commands)
            cmd_counter += 1
            brain.maybe_improve(every=20)

            # 6. Sleep on "stop"
            # (brain already handled intent==stop; check response text)
            if "sleep" in (response or "").lower() or success is False and "stop" in command:
                sleeping = True

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        print("\n[Agent] Shutting down …")
        perf_report = monitor.report()
        print(perf_report)
    finally:
        tts.speak("Agent offline. Goodbye.")
        logger.info("AI PC Agent stopped.")


if __name__ == "__main__":
    main()
