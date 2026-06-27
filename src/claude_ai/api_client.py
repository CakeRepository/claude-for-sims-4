"""
Claude API HTTP client.
Uses curl via subprocess since the Sims 4's embedded Python 3.7 lacks SSL support.
All calls are made on a background thread to avoid freezing the game.
"""
import datetime
import json
import os
import subprocess
import sys
import threading

from . import config, _log

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"

_LAST_PROMPT_FILENAME = "ClaudeAI_LastPrompt.txt"


def _last_prompt_path():
    """Path to the last-prompt log file (next to claude_config.cfg)."""
    cfg = config._find_config_file()
    if cfg:
        return os.path.join(os.path.dirname(cfg), _LAST_PROMPT_FILENAME)
    return os.path.join(os.path.expanduser("~"), "Documents", _LAST_PROMPT_FILENAME)


def _log_prompt(system, messages, model):
    """Write the most recent prompt to a file for debugging."""
    try:
        path = _last_prompt_path()
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== Claude AI — Last Prompt ===\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Model: {model}\n\n")
            f.write("=== SYSTEM PROMPT ===\n")
            f.write((system or "(none)") + "\n\n")
            f.write("=== USER MESSAGES ===\n")
            for m in messages:
                f.write(f"--- role: {m.get('role')} ---\n")
                f.write(str(m.get("content", "")) + "\n\n")
    except Exception:
        pass


def _extract_text(response_data):
    # Try native LM Studio stateful chat format first (output blocks)
    if "output" in response_data:
        try:
            output_list = response_data["output"]
            if isinstance(output_list, list):
                # Search for type: "message"
                for block in output_list:
                    if isinstance(block, dict) and block.get("type") == "message":
                        return block.get("content", "")
                # Fallback: if no type: "message", try the last block's content
                if len(output_list) > 0 and isinstance(output_list[-1], dict):
                    return output_list[-1].get("content", "")
        except Exception:
            pass

    # Try OpenAI / LM Studio format if choices is present
    if "choices" in response_data:
        try:
            return response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
    # Fallback to Anthropic format
    try:
        return response_data["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def call_claude_async(messages, system=None, use_fast_model=False, callback=None):
    """
    Make an async call to the Claude API (or LM Studio) on a background thread.

    Args:
        messages: list of {"role": "user"|"assistant", "content": str}
        system:   optional system prompt string
        use_fast_model: if True, uses fast_model from config (Haiku) instead of default
        callback: function(text: str | None, error: str | None) called when done

    Returns the background Thread object.
    """
    def _request():
        api_key = config.get_api_key()
        if not config.is_configured():
            if callback:
                callback(None, "No API key configured. Edit claude_config.cfg in your Mods folder.")
            return

        model = config.get_fast_model() if use_fast_model else config.get_default_model()
        max_tokens = config.get_max_tokens()
        use_lmstudio = config.get_use_lmstudio()
        lmstudio_url = config.get_lmstudio_api_url() if use_lmstudio else ""

        if use_lmstudio and "/api/v1/chat" in lmstudio_url:
            # Native stateful LM Studio formatting
            input_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    input_parts.append(f"User: {content}")
                else:
                    input_parts.append(f"Assistant: {content}")

            if len(messages) == 1 and messages[0].get("role") == "user":
                input_str = messages[0].get("content", "")
            else:
                input_str = "\n".join(input_parts)

            body = {
                "model": model,
                "system_prompt": system or "",
                "input": input_str
            }
        elif use_lmstudio:
            # OpenAI compatible formatting
            openai_messages = []
            if system:
                openai_messages.append({"role": "system", "content": system})
            openai_messages.extend(messages)
            body = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": openai_messages,
            }
        else:
            # Anthropic formatting
            body = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                body["system"] = system

        # Log the prompt so we can debug what Claude actually saw
        _log_prompt(system, messages, model)

        body_json = json.dumps(body)

        try:
            # Hide the terminal window on Windows. On Mac/Linux this attribute
            # doesn't exist on subprocess, so we pass None and rely on the
            # default behavior (subprocess on macOS doesn't pop a window anyway).
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0

            if use_lmstudio:
                url = lmstudio_url
                curl_cmd = [
                    "curl", "-s",
                    "-X", "POST",
                    "-H", "Content-Type: application/json",
                ]
                if api_key and api_key != "YOUR_API_KEY_HERE":
                    curl_cmd.extend(["-H", f"Authorization: Bearer {api_key}"])
            else:
                url = _API_URL
                curl_cmd = [
                    "curl", "-s",
                    "-X", "POST",
                    "-H", "Content-Type: application/json",
                    "-H", "x-api-key: " + api_key,
                    "-H", "anthropic-version: " + _API_VERSION,
                ]

            result = subprocess.run(
                curl_cmd + [
                    "-d", body_json,
                    url,
                ],
                capture_output=True,
                timeout=60,
                startupinfo=startupinfo,
            )

            stdout_str = (result.stdout or b"").decode("utf-8", errors="replace")
            stderr_str = (result.stderr or b"").decode("utf-8", errors="replace")

            if result.returncode != 0:
                err = stderr_str.strip() or f"curl exited with code {result.returncode}"
                if callback:
                    callback(None, f"Network error: {err}")
                return

            _log(f"  api_client: raw response: {stdout_str}")
            data = json.loads(stdout_str)

            # Check for API error response
            if "error" in data:
                msg = data["error"].get("message", str(data["error"]))
                if callback:
                    callback(None, f"API error: {msg}")
                return

            text = _extract_text(data)
            if callback:
                callback(text, None)

        except subprocess.TimeoutExpired:
            if callback:
                callback(None, "Request timed out after 60 seconds.")

        except json.JSONDecodeError:
            if callback:
                callback(None, f"Invalid response from API: {stdout_str[:200]}")

        except FileNotFoundError:
            if callback:
                callback(None, "curl not found. This mod requires Windows 10 or later.")

        except Exception as e:
            if callback:
                callback(None, f"Unexpected error: {type(e).__name__}: {e}")

    thread = threading.Thread(target=_request, daemon=True, name="ClaudeAI-Request")
    thread.start()
    return thread
