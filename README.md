# Jarvis AI – Voice Controlled PC Assistant

Control your entire PC with natural voice commands. Jarvis listens continuously for a wake word, interprets your command, and executes the correct OS action — all with voice feedback.

---

## Features

| Capability | Details |
|---|---|
| Wake word activation | "Jarvis", "Computer", "Assistant" (configurable) |
| Speech recognition | Google (online), Vosk (offline), Whisper (offline) |
| Text-to-speech | pyttsx3 – fully offline |
| System control | Volume, brightness, shutdown, restart, lock, screenshot |
| App control | Open / close Chrome, VS Code, Notepad, Spotify, and more |
| Browser control | Tabs, scroll, back/forward, Google search |
| File control | Open folders, search files, create notes |
| Command memory | "Repeat last command", history log |
| LLM intent parsing | Optional OpenAI fallback for natural language |

---

## Project Structure

```
jarvis_ai/
├── core/
│   ├── assistant_engine.py   # central orchestrator
│   └── intent_parser.py      # speech → (intent, value)
├── voice/
│   ├── speech_listener.py    # mic capture + STT
│   ├── wake_word.py          # wake word detection
│   └── tts_engine.py         # text-to-speech
├── control/
│   ├── system_control.py     # volume, brightness, power
│   ├── app_control.py        # open/close apps
│   ├── browser_control.py    # browser keyboard shortcuts
│   └── file_control.py       # folder/file operations
├── memory/
│   └── command_history.py    # command log + repeat
├── utils/
│   └── command_map.py        # all keywords, paths, templates
├── main.py
└── requirements.txt
```

---

## Setup

### 1. Python version

Requires **Python 3.10+**.

### 2. Install dependencies

```bash
pip install SpeechRecognition pyaudio pyautogui keyboard psutil pygetwindow pyttsx3 numpy
```

> **PyAudio on Windows:**  if `pip install pyaudio` fails, download the pre-built wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio and install with `pip install <wheel_file>.whl`.

### 3. (Optional) Offline STT – Vosk

```bash
pip install vosk
```

Download a model (e.g. `vosk-model-small-en-us-0.15`) from https://alphacephei.com/vosk/models and extract it to the project root.

### 4. (Optional) Offline STT – Whisper

```bash
pip install openai-whisper
```

### 5. Place a user photo (for ai_os_controller)

If you are also using the companion gesture controller, place a front-facing photo named `user.jpg` in the project root.

---

## Run

```bash
# From the workspace root (d:\websites\os controller)
python -m jarvis_ai.main
```

### Additional flags

```bash
# Disable wake-word gate (every utterance is treated as a command)
python -m jarvis_ai.main --no-wake

# Use Vosk for offline recognition
python -m jarvis_ai.main --backend vosk --vosk-model vosk-model-small-en-us-0.15

# Use Whisper
python -m jarvis_ai.main --backend whisper --whisper-model base

# Silent mode (no TTS, print responses only)
python -m jarvis_ai.main --silent

# Persist command history
python -m jarvis_ai.main --history-file jarvis_history.json

# Enable LLM natural-language intent parsing
python -m jarvis_ai.main --use-llm --openai-key sk-YOUR_KEY
```

---

## Voice Commands

Say the wake word first, then the command:

### System

| Say | Action |
|---|---|
| Jarvis volume up | Increase volume |
| Jarvis volume down | Decrease volume |
| Jarvis mute | Toggle mute |
| Jarvis brightness up | Increase display brightness |
| Jarvis brightness down | Decrease display brightness |
| Jarvis shutdown | Shut down the PC |
| Jarvis restart | Restart the PC |
| Jarvis lock screen | Lock the workstation |
| Jarvis screenshot | Save a screenshot |

### Applications

| Say | Action |
|---|---|
| Jarvis open chrome | Launch Chrome |
| Jarvis open vscode | Launch VS Code |
| Jarvis open notepad | Launch Notepad |
| Jarvis open spotify | Launch Spotify |
| Jarvis open calculator | Launch Calculator |
| Jarvis close chrome | Kill Chrome process |

### Browser

| Say | Action |
|---|---|
| Jarvis new tab | Ctrl+T |
| Jarvis close tab | Ctrl+W |
| Jarvis next tab | Ctrl+Tab |
| Jarvis scroll down | Scroll page down |
| Jarvis scroll up | Scroll page up |
| Jarvis go back | Alt+Left |
| Jarvis refresh | F5 |
| Jarvis search python tutorials | Google search |

### Files & Folders

| Say | Action |
|---|---|
| Jarvis open downloads | Open Downloads folder |
| Jarvis open documents | Open Documents folder |
| Jarvis open desktop | Open Desktop folder |
| Jarvis create file | Create a new text file on Desktop |
| Jarvis search file report | Open Windows search for "report" |

### Memory

| Say | Action |
|---|---|
| Jarvis repeat | Repeat the last successful command |
| Jarvis show history | List recent commands |
| Jarvis help | Hear a capabilities summary |
| Jarvis stop | Go to sleep (wake word re-activates) |

---

## Extending Jarvis

### Add a new app

Edit `utils/command_map.py`:

```python
APP_PATHS["myapp"] = r"C:\Path\To\myapp.exe"
APP_ALIASES["my application"] = "myapp"
```

### Add a new command keyword

Edit `INTENT_KEYWORDS` in `utils/command_map.py`:

```python
"my_intent": ["trigger phrase one", "trigger phrase two"],
```

Then handle `"my_intent"` in `core/assistant_engine.py` → `_dispatch()`.

### Switch to offline STT permanently

In `main.py` change the `--backend` default from `"google"` to `"vosk"` or `"whisper"`.

---

## Architecture

```
Microphone → SpeechListener → WakeWordDetector
                                     ↓
                             IntentParser
                                     ↓
                          AssistantEngine._dispatch()
                         /       |        |        \
               SystemControl  AppControl  Browser  FileControl
                                     ↓
                              TTSEngine (speak)
                                     ↓
                           CommandHistory (log)
```

---

## License

MIT – free to use, modify, and distribute.
