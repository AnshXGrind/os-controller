"""
utils/command_map.py

Central registry of:
    • intent → action-function name mappings
    • keyword → intent mappings  (used by the intent parser)
    • app name → executable path mappings
    • spoken aliases for common values

Keeping all keyword/path data here makes it trivial to extend
Jarvis without touching core logic files.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Keyword → Intent map
#
# Each entry is a list of trigger phrases.  The parser checks whether any
# of these phrases appears as a substring of the recognised utterance.
# Order within INTENT_KEYWORDS does NOT matter; specificity is achieved
# by checking longer phrases before shorter ones in the parser.
# ─────────────────────────────────────────────────────────────────────────────

INTENT_KEYWORDS: dict[str, list[str]] = {

    # ── App control ──────────────────────────────────────────────────────────
    "open_app":         ["open", "launch", "start", "run"],
    "close_app":        ["close", "kill", "exit", "quit", "stop"],

    # ── Volume ───────────────────────────────────────────────────────────────
    "volume_up":        ["volume up", "louder", "increase volume",
                         "turn up", "raise volume"],
    "volume_down":      ["volume down", "quieter", "decrease volume",
                         "turn down", "lower volume"],
    "mute":             ["mute", "silence", "quiet"],

    # ── Brightness ───────────────────────────────────────────────────────────
    "brightness_up":    ["brightness up", "brighter", "increase brightness"],
    "brightness_down":  ["brightness down", "dimmer", "decrease brightness"],

    # ── System ───────────────────────────────────────────────────────────────
    "shutdown":         ["shutdown", "shut down", "power off", "turn off"],
    "restart":          ["restart", "reboot", "restar"],
    "lock_screen":      ["lock screen", "lock", "sleep"],
    "screenshot":       ["screenshot", "screen shot", "capture screen"],

    # ── Browser ──────────────────────────────────────────────────────────────
    "new_tab":          ["new tab", "open tab"],
    "close_tab":        ["close tab"],
    "next_tab":         ["next tab", "switch tab right"],
    "previous_tab":     ["previous tab", "switch tab left", "prev tab"],
    "scroll_up":        ["scroll up", "go up", "page up"],
    "scroll_down":      ["scroll down", "go down", "page down"],
    "go_back":          ["go back", "back"],
    "go_forward":       ["go forward", "forward"],
    "refresh":          ["refresh", "reload"],
    "search_google":    ["search", "google", "look up", "find"],

    # ── Window management ────────────────────────────────────────────────────
    "minimize_window":  ["minimize", "minimise"],
    "maximize_window":  ["maximize", "maximise", "full screen", "fullscreen"],
    "close_window":     ["close window"],
    "switch_window":    ["switch window", "alt tab", "next window"],

    # ── File control ─────────────────────────────────────────────────────────
    "open_folder":      ["open folder", "open documents", "open downloads",
                         "open desktop"],
    "create_file":      ["create file", "new file", "make file"],
    "search_file":      ["search file", "find file", "locate file"],

    # ── Memory / meta ────────────────────────────────────────────────────────
    "repeat_command":   ["repeat", "do that again", "again"],
    "show_history":     ["show history", "what did i say"],
    "help":             ["help", "what can you do", "commands"],
    "stop":             ["stop listening", "go to sleep", "goodbye", "bye"],
}


# ─────────────────────────────────────────────────────────────────────────────
# App name → executable path
#
# Paths use environment variables so they adapt to different users/machines.
# ─────────────────────────────────────────────────────────────────────────────

_APPDATA      = os.environ.get("APPDATA",      "C:\\Users\\User\\AppData\\Roaming")
_LOCAL        = os.environ.get("LOCALAPPDATA", "C:\\Users\\User\\AppData\\Local")
_PROGRAMFILES = os.environ.get("PROGRAMFILES", "C:\\Program Files")
_PF86         = os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)")
_USERPROFILE  = os.environ.get("USERPROFILE",  "C:\\Users\\User")

APP_PATHS: dict[str, str] = {
    "chrome":      rf"{_PROGRAMFILES}\Google\Chrome\Application\chrome.exe",
    "firefox":     rf"{_PROGRAMFILES}\Mozilla Firefox\firefox.exe",
    "edge":        rf"{_PROGRAMFILES}\Microsoft\Edge\Application\msedge.exe",
    "vscode":      rf"{_LOCAL}\Programs\Microsoft VS Code\Code.exe",
    "notepad":     "notepad.exe",
    "wordpad":     "wordpad.exe",
    "calculator":  "calc.exe",
    "explorer":    "explorer.exe",
    "paint":       "mspaint.exe",
    "cmd":         "cmd.exe",
    "powershell":  "powershell.exe",
    "spotify":     rf"{_APPDATA}\Spotify\Spotify.exe",
    "discord":     rf"{_LOCAL}\Discord\Update.exe",
    "vlc":         rf"{_PROGRAMFILES}\VideoLAN\VLC\vlc.exe",
    "steam":       rf"{_PROGRAMFILES(86)}\Steam\steam.exe"
                   if (x := rf"{_PF86}\Steam\steam.exe") else x,
    "task manager":"taskmgr.exe",
}

# Spoken aliases → canonical app key (e.g. "open google" → chrome)
APP_ALIASES: dict[str, str] = {
    "google":    "chrome",
    "browser":   "chrome",
    "vs code":   "vscode",
    "code":      "vscode",
    "text editor":"notepad",
    "music":     "spotify",
    "videos":    "vlc",
    "files":     "explorer",
    "terminal":  "powershell",
}


# ─────────────────────────────────────────────────────────────────────────────
# Common folder shortcuts → Windows shell paths
# ─────────────────────────────────────────────────────────────────────────────

FOLDER_PATHS: dict[str, str] = {
    "desktop":   rf"{_USERPROFILE}\Desktop",
    "documents": rf"{_USERPROFILE}\Documents",
    "downloads": rf"{_USERPROFILE}\Downloads",
    "pictures":  rf"{_USERPROFILE}\Pictures",
    "music":     rf"{_USERPROFILE}\Music",
    "videos":    rf"{_USERPROFILE}\Videos",
}


# ─────────────────────────────────────────────────────────────────────────────
# Intent → human-readable response template
# ─────────────────────────────────────────────────────────────────────────────

RESPONSE_TEMPLATES: dict[str, str] = {
    "open_app":        "Opening {value}",
    "close_app":       "Closing {value}",
    "volume_up":       "Turning up the volume",
    "volume_down":     "Turning down the volume",
    "mute":            "Muting audio",
    "brightness_up":   "Increasing brightness",
    "brightness_down": "Decreasing brightness",
    "shutdown":        "Shutting down the system",
    "restart":         "Restarting the system",
    "lock_screen":     "Locking the screen",
    "screenshot":      "Taking a screenshot",
    "new_tab":         "Opening a new tab",
    "close_tab":       "Closing the tab",
    "next_tab":        "Switching to the next tab",
    "previous_tab":    "Switching to the previous tab",
    "scroll_up":       "Scrolling up",
    "scroll_down":     "Scrolling down",
    "go_back":         "Going back",
    "go_forward":      "Going forward",
    "refresh":         "Refreshing the page",
    "search_google":   "Searching for {value}",
    "minimize_window": "Minimizing the window",
    "maximize_window": "Maximizing the window",
    "close_window":    "Closing the window",
    "switch_window":   "Switching windows",
    "open_folder":     "Opening {value}",
    "create_file":     "Creating a new file",
    "search_file":     "Searching for {value}",
    "repeat_command":  "Repeating the last command",
    "show_history":    "Here are your recent commands",
    "help":            "Here is what I can do for you",
    "stop":            "Going to sleep. Goodbye!",
    "unknown":         "Sorry, I didn't understand that command",
}
