"""Graphical launcher for RichBich Telegram bot."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue

import tkinter as tk
from tkinter import messagebox, ttk

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LauncherApp:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("RichBich Launcher")
        self.master.geometry("780x560")
        self.master.minsize(700, 520)

        self.token_var = tk.StringVar(value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
        self.provider_var = tk.StringVar(value=os.getenv("LLM_PROVIDER", "openai"))
        self.llm_key_var = tk.StringVar(value=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")))
        self.llm_model_var = tk.StringVar(value=os.getenv("LLM_MODEL", "gpt-4o-mini"))
        self.llm_base_url_var = tk.StringVar(value=os.getenv("LLM_BASE_URL", ""))
        self.config_var = tk.StringVar(value=str(PROJECT_ROOT / "configs" / "training.yaml"))
        self.model_path_var = tk.StringVar(value="")

        self.bot_process: subprocess.Popen[str] | None = None
        self.train_process: subprocess.Popen[str] | None = None
        self._temp_configs: list[Path] = []

        self.log_queue: Queue[str] = Queue()
        self._log_after_id: str | None = None

        self._build_ui()
        self._start_log_pump()

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def _bind_context_menu(self, widget: tk.Widget) -> None:
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Cut", command=lambda: widget.event_generate('<<Cut>>'))
        menu.add_command(label="Copy", command=lambda: widget.event_generate('<<Copy>>'))
        menu.add_command(label="Paste", command=lambda: widget.event_generate('<<Paste>>'))
        menu.add_separator()
        menu.add_command(label="Select All", command=lambda: widget.event_generate('<<SelectAll>>'))

        def _show_menu(event: tk.Event) -> None:
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        def _select_all(event: tk.Event) -> str:
            widget.event_generate('<<SelectAll>>')
            return 'break'

        widget.bind('<Button-3>', _show_menu, add='+')
        widget.bind('<Control-a>', _select_all, add='+')

    def _build_ui(self) -> None:
        container = ttk.Frame(self.master, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        for idx in range(3):
            container.columnconfigure(idx, weight=1 if idx == 1 else 0)

        self._add_labeled_entry(container, "Telegram токен", self.token_var, row=0)
        self._add_labeled_combobox(
            container,
            "LLM провайдер",
            self.provider_var,
            values=("openai", "openai-compatible", "azure", "ollama", "none"),
            row=1,
        )
        self._add_labeled_entry(container, "LLM API ключ", self.llm_key_var, row=2, show="*")
        self._add_labeled_entry(container, "LLM модель", self.llm_model_var, row=3)
        self._add_labeled_entry(container, "LLM base URL", self.llm_base_url_var, row=4)
        self._add_labeled_entry(container, "Конфиг", self.config_var, row=5)
        self._add_labeled_entry(container, "Модель (опция)", self.model_path_var, row=6)

        button_row = ttk.Frame(container)
        button_row.grid(column=0, columnspan=3, row=7, pady=(12, 8), sticky="ew")
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        button_row.columnconfigure(2, weight=1)

        self.start_button = ttk.Button(button_row, text="Запустить бота", command=self.start_bot)
        self.start_button.grid(column=0, row=0, padx=4, sticky="ew")

        self.stop_button = ttk.Button(button_row, text="Остановить бота", command=self.stop_bot)
        self.stop_button.grid(column=1, row=0, padx=4, sticky="ew")

        self.train_button = ttk.Button(button_row, text="Обучить (3 месяца)", command=self.train_model)
        self.train_button.grid(column=2, row=0, padx=4, sticky="ew")

        log_label = ttk.Label(container, text="Журнал событий", anchor="w")
        log_label.grid(column=0, row=8, columnspan=3, sticky="ew")

        self.log_text = tk.Text(container, height=18, wrap="word", state=tk.DISABLED)
        self.log_text.grid(column=0, row=9, columnspan=3, sticky="nsew")
        container.rowconfigure(9, weight=1)
        self._bind_context_menu(self.log_text)

        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(column=3, row=9, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        *,
        row: int,
        show: str | None = None,
    ) -> None:
        ttk.Label(parent, text=label, anchor="w").grid(column=0, row=row, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=variable, show=show or "")
        entry.grid(column=1, row=row, columnspan=2, sticky="ew", padx=(8, 0))
        self._bind_context_menu(entry)

    def _add_labeled_combobox(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        *,
        values: tuple[str, ...],
        row: int,
    ) -> None:
        ttk.Label(parent, text=label, anchor="w").grid(column=0, row=row, sticky="w", pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(column=1, row=row, columnspan=2, sticky="ew", padx=(8, 0))
        combo.set(variable.get() if variable.get() in values else values[0])
        self._bind_context_menu(combo)

    def _start_log_pump(self) -> None:
        if self._log_after_id is None:
            self._log_after_id = self.master.after(200, self._drain_log_queue)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
        except Empty:
            pass
        self._log_after_id = self.master.after(300, self._drain_log_queue)

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {line}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _reader_thread(self, process: subprocess.Popen[str], label: str) -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            self.log_queue.put(f"{label}: {raw.rstrip()}")
        code = process.wait()
        self.log_queue.put(f"{label}: процесс завершён с кодом {code}")

    def start_bot(self) -> None:
        if self.bot_process and self.bot_process.poll() is None:
            messagebox.showinfo("RichBich", "Бот уже запущен.")
            return

        token = self.token_var.get().strip()
        if not token:
            messagebox.showerror("RichBich", "Укажите Telegram токен.")
            return

        config_path = Path(self.config_var.get().strip())
        if not config_path.exists():
            messagebox.showerror("RichBich", f"Конфиг {config_path} не найден.")
            return

        env = os.environ.copy()
        env["TELEGRAM_BOT_TOKEN"] = token

        provider = self.provider_var.get().strip().lower()
        env["LLM_PROVIDER"] = provider

        api_key = self.llm_key_var.get().strip()
        if api_key:
            env["LLM_API_KEY"] = api_key
            if provider in {"openai", "openai-compatible", "azure"}:
                env.setdefault("OPENAI_API_KEY", api_key)
        else:
            env.pop("LLM_API_KEY", None)

        model = self.llm_model_var.get().strip()
        if model:
            env["LLM_MODEL"] = model
        else:
            env.pop("LLM_MODEL", None)

        base_url = self.llm_base_url_var.get().strip()
        if base_url:
            env["LLM_BASE_URL"] = base_url
        else:
            env.pop("LLM_BASE_URL", None)

        cmd = [sys.executable, "scripts/start_bot.py", "--config", str(config_path)]
        model_path = self.model_path_var.get().strip()
        if model_path:
            cmd.extend(["--model-path", model_path])

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags,
            )
        except FileNotFoundError as exc:
            messagebox.showerror("RichBich", f"Не удалось запустить бота: {exc}")
            return

        self.bot_process = process
        threading.Thread(target=self._reader_thread, args=(process, "bot"), daemon=True).start()
        self._append_log("bot: запуск Telegram бота инициирован")

    def stop_bot(self) -> None:
        if not self.bot_process or self.bot_process.poll() is not None:
            messagebox.showinfo("RichBich", "Бот не запущен.")
            return
        self.bot_process.terminate()
        self._append_log("bot: отправлен сигнал остановки")

    def train_model(self) -> None:
        if self.train_process and self.train_process.poll() is None:
            messagebox.showinfo("RichBich", "Обучение уже выполняется.")
            return

        config_path = Path(self.config_var.get().strip())
        if not config_path.exists():
            messagebox.showerror("RichBich", f"Конфиг {config_path} не найден.")
            return

        try:
            quick_config = self._prepare_quick_config(config_path)
        except Exception as exc:
            messagebox.showerror("RichBich", f"Не удалось подготовить конфиг: {exc}")
            return

        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(quick_config),
            "--print-metrics",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        self.train_process = process
        threading.Thread(target=self._reader_thread, args=(process, "train"), daemon=True).start()
        self._append_log(f"train: обучение запущено с конфигом {quick_config}")

    def _prepare_quick_config(self, base_config: Path) -> Path:
        data = yaml.safe_load(base_config.read_text(encoding="utf-8"))
        now = datetime.utcnow().date()
        start = now - timedelta(days=90)

        data.setdefault("data", {})
        data["data"]["start_date"] = start.isoformat()
        data["data"]["end_date"] = now.isoformat()

        data.setdefault("news", {})
        data["news"]["start_date"] = start.isoformat()
        data["news"]["end_date"] = now.isoformat()

        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f"richbich_quick_{uuid.uuid4().hex}.yaml"
        temp_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        self._temp_configs.append(temp_path)
        return temp_path

    def on_close(self) -> None:
        for proc in (self.bot_process, self.train_process):
            if proc and proc.poll() is None:
                proc.terminate()
        if self._log_after_id is not None:
            try:
                self.master.after_cancel(self._log_after_id)
            except Exception:
                pass
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except Empty:
                break
        for temp in self._temp_configs:
            try:
                temp.unlink(missing_ok=True)
            except Exception:
                pass
        self.master.destroy()


def main() -> None:
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
