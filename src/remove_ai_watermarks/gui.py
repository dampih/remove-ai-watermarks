"""Remove-AI-Watermarks GUI — Qt6 desktop application.

Provides a tabbed interface for all major operations:
  - Single-image visible watermark removal
  - Full pipeline (visible + invisible + metadata)
  - Batch directory processing
  - Provenance identification
  - Region eraser
  - Video processing
  - Metadata operations
"""

# PyQt6's generated stubs and optional cv2/numpy imports expose dynamic members.
# Keep strict checking for the rest of the package while allowing those bindings here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIncompatibleMethodOverride=false, reportUntypedFunctionDecorator=false, reportConstantRedefinition=false

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


# ── Qt imports ─────────────────────────────────────────────────────────────
from PyQt6.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ── Optional imports ───────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    CV2_AVAILABLE = False


# ── Background task runner ──────────────────────────────────────────────────


class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)


class Worker(QRunnable):
    """Run a callable in a background thread."""

    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__()
        self._fn = fn
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._fn()
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


# ── Thumbnail / preview helpers ─────────────────────────────────────────────


def _cv2_to_qpixmap(bgr: Any, max_size: int = 640) -> QPixmap:
    """Convert a cv2 BGR array to a QPixmap, fitting within max_size."""
    if bgr is None or not CV2_AVAILABLE:
        return QPixmap()
    h, w = bgr.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h2, w2 = rgb.shape[:2]
    img = QImage(rgb.data, w2, h2, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


def _load_cv2(path: str | Path) -> Any:
    """Load an image as cv2 BGR array."""
    if not CV2_AVAILABLE:
        return None
    return cv2.imread(str(path))


# ── Drop-zone widget ────────────────────────────────────────────────────────


class DropZone(QFrame):
    """A drag-and-drop target that accepts image/video files."""

    files_dropped = pyqtSignal(list)

    def __init__(self, accept_multiple: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._accept_multiple = accept_multiple
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "DropZone { border: 2px dashed #aaa; border-radius: 8px; "
            "background: #f8f8f8; padding: 20px; }"
            "DropZone:hover { border-color: #4a90d9; background: #eef6ff; }"
        )
        self._label = QLabel("Drag & drop file(s)\nor click to browse", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #666; font-size: 14px; border: none;")
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        self._paths: list[Path] = []

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None:
            return
        urls = event.mimeData().urls()
        self._paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        if not self._accept_multiple:
            self._paths = self._paths[:1]
        self.files_dropped.emit(self._paths)
        self._update_label()

    def mousePressEvent(self, event: Any) -> None:  # type: ignore[override]
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select files",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.heic *.avif);;"
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;"
            "All files (*.*)",
        )
        if paths:
            self._paths = [Path(p) for p in paths]
            self.files_dropped.emit(self._paths)
            self._update_label()

    def clear(self) -> None:
        self._paths.clear()
        self._update_label()

    def _update_label(self) -> None:
        n = len(self._paths)
        if n == 0:
            self._label.setText("Drag & drop file(s)\nor click to browse")
        else:
            names = ",\n".join(p.name for p in self._paths[:3])
            more = f"\n... and {n - 3} more" if n > 3 else ""
            self._label.setText(f"Loaded {n} file(s):\n{names}{more}")


# ── Log panel ───────────────────────────────────────────────────────────────


class LogPanel(QPlainTextEdit):
    """Scrolling log output."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont("Consolas, Courier New, monospace")
        font.setPointSize(10)
        self.setFont(font)
        self.setMaximumBlockCount(500)
        self.setStyleSheet("background: #1e1e1e; color: #d4d4d4;")

    def log(self, msg: str) -> None:
        self.appendPlainText(msg)

    def error(self, msg: str) -> None:
        self.appendHtml(f'<span style="color:#f48771;"> \u274c {msg}</span>')

    def success(self, msg: str) -> None:
        self.appendHtml(f'<span style="color:#6a9955;"> \u2713 {msg}</span>')

    def info(self, msg: str) -> None:
        self.appendHtml(f'<span style="color:#569cd6;"> \u2139 {msg}</span>')


# ── Preview panel ───────────────────────────────────────────────────────────


class PreviewPanel(QScrollArea):
    """Before/after image preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        content = QWidget()
        self.setWidget(content)
        layout = QHBoxLayout(content)

        self._before_label = QLabel("No image loaded")
        self._before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._before_label.setStyleSheet(
            "color: #888; border: 1px solid #ddd; border-radius: 4px; padding: 10px;"
        )

        self._after_label = QLabel("No result yet")
        self._after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._after_label.setStyleSheet(
            "color: #888; border: 1px solid #ddd; border-radius: 4px; padding: 10px;"
        )

        for lbl in (self._before_label, self._after_label):
            lbl.setMinimumSize(300, 200)
            lbl.setScaledContents(True)

        layout.addWidget(self._before_label, 1)
        layout.addWidget(self._after_label, 1)

    def show_before(self, bgr: Any) -> None:
        pix = _cv2_to_qpixmap(bgr)
        if pix.isNull():
            self._before_label.setText("Cannot preview")
        else:
            self._before_label.setPixmap(pix)
            self._before_label.setText("")

    def show_after(self, bgr: Any) -> None:
        pix = _cv2_to_qpixmap(bgr)
        if pix.isNull():
            self._after_label.setText("No result")
        else:
            self._after_label.setPixmap(pix)
            self._after_label.setText("")

    def clear(self) -> None:
        self._before_label.clear()
        self._before_label.setText("No image loaded")
        self._after_label.clear()
        self._after_label.setText("No result yet")


# ── Mode widgets ────────────────────────────────────────────────────────────


class RemovalTab(QWidget):
    """Single-image visible watermark removal."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_bgr: Any = None
        self._result_bgr: Any = None
        self._path: Path | None = None

        layout = QVBoxLayout(self)

        # Top controls
        ctrl = QHBoxLayout()
        self.dropzone = DropZone(accept_multiple=False)
        self.dropzone.setMinimumHeight(100)
        self.dropzone.files_dropped.connect(self._on_files)

        opts = QGroupBox("Settings")
        oform = QFormLayout(opts)
        self.mark_cb = QComboBox()
        self.mark_cb.addItems(
            ["auto", "gemini", "doubao", "jimeng", "qwen", "kling",
             "yuanbao", "samsung", "runninghub", "baidu", "liblib",
             "microsoft", "dola", "jimeng_pill"]
        )
        oform.addRow("Mark:", self.mark_cb)

        self.sensitivity_cb = QComboBox()
        self.sensitivity_cb.addItems(["auto", "strict"])
        oform.addRow("Sensitivity:", self.sensitivity_cb)

        self.backend_cb = QComboBox()
        self.backend_cb.addItems(["auto", "cv2", "migan", "lama"])
        oform.addRow("Backend:", self.backend_cb)

        self.strip_md_cb = QCheckBox("Strip AI metadata")
        self.strip_md_cb.setChecked(True)
        oform.addRow(self.strip_md_cb)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25b6 Remove Watermark")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(
            "QPushButton { background: #4a90d9; color: white; padding: 8px 20px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:disabled { background: #ccc; color: #666; }"
        )
        self.save_btn = QPushButton("\U0001f4be Save As...")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()

        # Preview
        self.preview = PreviewPanel()
        self.log = LogPanel()
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.preview)
        splitter.addWidget(self.log)

        layout.addLayout(ctrl)
        ctrl.addWidget(self.dropzone, 1)
        ctrl.addWidget(opts)
        layout.addLayout(btn_row)
        layout.addWidget(self.progress)
        layout.addWidget(splitter, 1)

    def _on_files(self, paths: list[Path]) -> None:
        if not paths:
            return
        self._path = paths[0]
        self._current_bgr = _load_cv2(self._path)
        if self._current_bgr is not None:
            self.preview.show_before(self._current_bgr)
            self.run_btn.setEnabled(True)
            self.log.info(
                f"Loaded: {self._path.name} ({self._current_bgr.shape[1]}x{self._current_bgr.shape[0]})"
            )
        else:
            self.log.error(f"Cannot read image: {self._path.name}")

    def _run(self) -> None:
        if self._current_bgr is None:
            return
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate
        self.log.info("Running visible watermark removal...")

        mark = self.mark_cb.currentText()
        sensitivity = self.sensitivity_cb.currentText()  # type: ignore[assignment]
        backend = self.backend_cb.currentText()  # type: ignore[assignment]
        bgr_copy = self._current_bgr.copy()

        def _work() -> object:
            from remove_ai_watermarks import api as raiw_api
            from remove_ai_watermarks import watermark_registry as wr

            if mark == "auto":
                result, removed = raiw_api.remove_visible(
                    bgr_copy,
                    sensitivity=sensitivity,
                    backend=backend,
                    strip_metadata=False,
                )
            else:
                # Single explicit mark
                img = bgr_copy.copy()
                removed = []
                km = wr.get_mark(mark)
                det = km.detect(img, provenance=False)
                if det.detected:
                    mask = km.mask(img, det)
                    if mask is not None:
                        img = wr.fill(img, mask, backend=backend)
                    removed.append(mark)
                result = img
            return result, removed

        worker = Worker(_work)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self.log.error)
        worker.signals.finished.connect(self._on_finished)
        pool = QThreadPool.globalInstance()
        if pool is not None:
            pool.start(worker)

    def _on_finished(self) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _on_result(self, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            self.log.error("Unexpected result from removal")
            return
        self._result_bgr, removed = result
        if isinstance(removed, list) and removed:
            self.log.success("Removed: {}".format(", ".join(removed)))
            self.preview.show_after(self._result_bgr)
            self.save_btn.setEnabled(True)
        else:
            self.log.info("No known visible watermark detected.")

    def _save(self) -> None:
        if self._result_bgr is None:
            return
        default_name = f"{self._path.stem}_clean{self._path.suffix}" if self._path else "output.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save cleaned image", default_name,
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path and CV2_AVAILABLE:
            cv2.imwrite(str(path), self._result_bgr)
            self.log.success(f"Saved: {path}")


class FullCleanTab(QWidget):
    """Full pipeline: visible + invisible + metadata."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._path: Path | None = None

        ctrl = QHBoxLayout()
        self.dropzone = DropZone(accept_multiple=False)
        self.dropzone.setMinimumHeight(80)
        self.dropzone.files_dropped.connect(self._on_files)

        opts = QGroupBox("Pipeline options")
        oform = QFormLayout(opts)
        self.force_cb = QCheckBox("Force diffusion (even without detected signal)")
        oform.addRow(self.force_cb)
        self.backend_cb = QComboBox()
        self.backend_cb.addItems(["auto", "cv2", "migan", "lama"])
        oform.addRow("Inpaint backend:", self.backend_cb)
        self.sensitivity_cb = QComboBox()
        self.sensitivity_cb.addItems(["auto", "strict"])
        oform.addRow("Sensitivity:", self.sensitivity_cb)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25b6 Run Full Pipeline")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(
            "background: #d94a4a; color: white; padding: 8px 20px; "
            "border-radius: 4px; font-weight: bold;"
        )
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()

        self.log = LogPanel()
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        layout.addLayout(ctrl)
        ctrl.addWidget(self.dropzone, 1)
        ctrl.addWidget(opts)
        layout.addLayout(btn_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.log, 1)

    def _on_files(self, paths: list[Path]) -> None:
        if paths:
            self._path = paths[0]
            self.run_btn.setEnabled(True)
            self.log.info(f"Loaded: {self._path.name}")

    def _run(self) -> None:
        if self._path is None:
            return
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log.info("Running full pipeline (visible \u2192 invisible \u2192 metadata)...")

        src = self._path
        backend = self.backend_cb.currentText()  # type: ignore[assignment]
        sensitivity = self.sensitivity_cb.currentText()  # type: ignore[assignment]
        force = self.force_cb.isChecked()

        def _work() -> object:
            from remove_ai_watermarks import api as raiw_api
            from remove_ai_watermarks.api import InvisibleOptions

            suffix = src.suffix or ".png"
            fd, outpath = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            return raiw_api.remove_all(
                src, outpath,
                backend=backend,
                sensitivity=sensitivity,
                force=force,
                invisible=InvisibleOptions(),
            )

        worker = Worker(_work)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)
        worker.signals.finished.connect(self._on_finished)
        pool = QThreadPool.globalInstance()
        if pool is not None:
            pool.start(worker)

    def _on_finished(self) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _on_result(self, result: object) -> None:
        from remove_ai_watermarks.api import RemoveAllResult
        if isinstance(result, RemoveAllResult):
            if result.visible_label:
                self.log.info(f"Visible marks: {result.visible_label}")
            self.log.info(f"Invisible: {result.invisible}")
            self.log.success(f"Pipeline complete \u2192 {result.output}")
        else:
            self.log.info("Pipeline returned unexpected type.")

    def _on_error(self, err: str) -> None:
        self.log.error(err)


class BatchTab(QWidget):
    """Batch directory processing."""

    MODES: list[str] = ["visible", "invisible", "metadata", "all"]  # noqa: RUF012

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        dirs = QHBoxLayout()
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("Source directory...")
        self.src_btn = QPushButton("Browse...")
        self.src_btn.clicked.connect(lambda: self._browse_dir(self.src_edit))
        self.dst_edit = QLineEdit()
        self.dst_edit.setPlaceholderText("Output directory (same = in-place)...")
        self.dst_btn = QPushButton("Browse...")
        self.dst_btn.clicked.connect(lambda: self._browse_dir(self.dst_edit))
        dirs.addWidget(QLabel("Source:"))
        dirs.addWidget(self.src_edit, 1)
        dirs.addWidget(self.src_btn)
        dirs.addWidget(QLabel("Output:"))
        dirs.addWidget(self.dst_edit, 1)
        dirs.addWidget(self.dst_btn)

        opts = QHBoxLayout()
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(self.MODES)
        opts.addWidget(QLabel("Mode:"))
        opts.addWidget(self.mode_cb)
        self.backend_cb = QComboBox()
        self.backend_cb.addItems(["auto", "cv2", "migan", "lama"])
        opts.addWidget(QLabel("Backend:"))
        opts.addWidget(self.backend_cb)
        opts.addStretch()

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25b6 Run Batch")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(
            "background: #d94a4a; color: white; padding: 8px 20px; "
            "border-radius: 4px; font-weight: bold;"
        )
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()

        self.log = LogPanel()
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        layout.addLayout(dirs)
        layout.addLayout(opts)
        layout.addLayout(btn_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.log, 1)

        self.src_edit.textChanged.connect(self._check_ready)
        self.dst_edit.textChanged.connect(self._check_ready)

    def _browse_dir(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select directory")
        if d:
            edit.setText(d)

    def _check_ready(self) -> None:
        self.run_btn.setEnabled(bool(self.src_edit.text()))

    def _run(self) -> None:
        src = Path(self.src_edit.text())
        dst = Path(self.dst_edit.text()) if self.dst_edit.text() else src
        mode = self.mode_cb.currentText()
        backend = self.backend_cb.currentText()

        if not src.is_dir():
            self.log.error(f"Source is not a directory: {src}")
            return

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log.info(f"Batch {mode} \u2192 src={src} dst={dst}")

        def _work() -> object:
            import warnings as _warnings
            _warnings.filterwarnings("ignore")
            from remove_ai_watermarks import api as raiw_api
            from remove_ai_watermarks.api import InvisibleOptions

            return raiw_api.remove_batch(
                src, dst,
                mode=mode,  # type: ignore[arg-type]
                backend=backend,  # type: ignore[arg-type]
                sensitivity="auto",
                invisible=InvisibleOptions(),
            )

        worker = Worker(_work)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self.log.error)
        worker.signals.finished.connect(self._on_finished)
        pool = QThreadPool.globalInstance()
        if pool is not None:
            pool.start(worker)

    def _on_finished(self) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _on_result(self, result: object) -> None:
        from remove_ai_watermarks.api import BatchSummary
        if isinstance(result, BatchSummary):
            self.log.success(f"Processed: {result.processed}, failed: {result.failed}")
            if hasattr(result, "unavailable") and result.unavailable:
                self.log.info(
                    f"Unavailable (no signal): {len(result.unavailable)} files"
                )  # type: ignore[arg-type,union-attr]
            for p, err in result.errors:
                self.log.error(f"  {p.name}: {err}")
        else:
            self.log.info("Batch done (unknown result type).")


class IdentifyTab(QWidget):
    """Provenance identification."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.dropzone = DropZone(accept_multiple=True)
        self.dropzone.setMinimumHeight(80)
        self.dropzone.files_dropped.connect(self._on_files)
        self._last_paths: list[Path] = []

        self.run_btn = QPushButton("\U0001f50d Identify")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("padding: 8px 20px; border-radius: 4px; font-weight: bold;")

        self.result_tree = QTextEdit()
        self.result_tree.setReadOnly(True)
        self.result_tree.setStyleSheet(
            "background: #1e1e1e; color: #d4d4d4; font-family: Consolas;"
        )

        layout.addWidget(self.dropzone)
        layout.addWidget(self.run_btn)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.result_tree, 1)

    def _on_files(self, paths: list[Path]) -> None:
        self._last_paths = paths
        self.run_btn.setEnabled(len(paths) > 0)

    def _run(self) -> None:
        self.run_btn.setEnabled(False)
        self.result_tree.clear()

        paths = list(self._last_paths)

        def _work() -> object:
            from remove_ai_watermarks.identify import identify

            reports: dict[str, Any] = {}
            for p in paths:
                try:
                    rep = identify(p)
                    reports[p.name] = rep.to_dict()
                except Exception as e:
                    reports[p.name] = {"error": str(e)}
            return reports

        worker = Worker(_work)
        worker.signals.result.connect(self._show_reports)
        worker.signals.error.connect(self.result_tree.setPlainText)
        worker.signals.finished.connect(lambda: self.run_btn.setEnabled(True))
        pool = QThreadPool.globalInstance()
        if pool is not None:
            pool.start(worker)

    def _show_reports(self, reports: object) -> None:
        import json
        if isinstance(reports, dict):
            text = json.dumps(reports, indent=2, ensure_ascii=False)
            self.result_tree.setPlainText(text)
        self.run_btn.setEnabled(True)


class EraseTab(QWidget):
    """Region eraser."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._path: Path | None = None
        self._bgr: Any = None
        self._result: Any = None

        self.dropzone = DropZone(accept_multiple=False)
        self.dropzone.setMinimumHeight(80)
        self.dropzone.files_dropped.connect(self._on_files)

        region_box = QGroupBox("Region")
        rform = QFormLayout(region_box)
        self.region_x = QSpinBox(maximum=10000)
        self.region_y = QSpinBox(maximum=10000)
        self.region_w = QSpinBox(maximum=10000, value=100)
        self.region_h = QSpinBox(maximum=10000, value=100)
        rform.addRow("X:", self.region_x)
        rform.addRow("Y:", self.region_y)
        rform.addRow("Width:", self.region_w)
        rform.addRow("Height:", self.region_h)

        opts = QHBoxLayout()
        self.backend_cb = QComboBox()
        self.backend_cb.addItems(["cv2", "migan", "lama"])
        self.backend_cb.setCurrentText("cv2")
        opts.addWidget(QLabel("Backend:"))
        opts.addWidget(self.backend_cb)
        opts.addStretch()

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25b6 Erase Region")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.save_btn = QPushButton("\U0001f4be Save As...")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()

        self.preview = PreviewPanel()
        self.log = LogPanel()
        layout.addWidget(self.dropzone)
        layout.addWidget(region_box)
        layout.addLayout(opts)
        layout.addLayout(btn_row)
        layout.addWidget(self.preview)
        layout.addWidget(self.log, 1)

    def _on_files(self, paths: list[Path]) -> None:
        if not paths:
            return
        self._path = paths[0]
        self._bgr = _load_cv2(self._path)
        if self._bgr is not None:
            self.preview.show_before(self._bgr)
            self.region_x.setMaximum(self._bgr.shape[1])
            self.region_y.setMaximum(self._bgr.shape[0])
            self.run_btn.setEnabled(True)
            self.log.info(f"Loaded: {self._path.name}")

    def _run(self) -> None:
        if self._bgr is None:
            return
        x, y, w, h = (
            self.region_x.value(),
            self.region_y.value(),
            self.region_w.value(),
            self.region_h.value(),
        )
        backend = self.backend_cb.currentText()
        self.log.info(f"Erasing region ({x},{y},{w}x{h}) with backend={backend}...")

        bgr_copy = self._bgr.copy()

        def _work() -> object:
            import numpy as np

            from remove_ai_watermarks.watermark_registry import fill

            mask = np.zeros(bgr_copy.shape[:2], np.uint8)
            mask[y:y + h, x:x + w] = 255
            return fill(bgr_copy, mask, backend=backend)  # type: ignore[arg-type]

        worker = Worker(_work)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self.log.error)
        worker.signals.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.run_btn.setEnabled(False)
        pool = QThreadPool.globalInstance()
        if pool is not None:
            pool.start(worker)

    def _on_result(self, result: object) -> None:
        if np is not None and isinstance(result, np.ndarray):
            self._result = result
            self.preview.show_after(self._result)
            self.save_btn.setEnabled(True)
            self.log.success("Region erased.")
        else:
            self.log.error("Unexpected result")

    def _save(self) -> None:
        if self._result is None:
            return
        default = f"{self._path.stem}_erased{self._path.suffix}" if self._path else "output.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save", default, "Images (*.png *.jpg *.webp)"
        )
        if path and CV2_AVAILABLE:
            cv2.imwrite(str(path), self._result)
            self.log.success(f"Saved: {path}")


class VideoTab(QWidget):
    """Video operations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._path: Path | None = None

        self.dropzone = DropZone(accept_multiple=False)
        self.dropzone.setMinimumHeight(80)
        self.dropzone.files_dropped.connect(self._on_files)

        opts = QHBoxLayout()
        self.action_cb = QComboBox()
        self.action_cb.addItems(
            ["Identify", "Remove visible marks", "Remove metadata", "Remove all"]
        )
        self.action_cb.currentTextChanged.connect(self._on_action_changed)
        opts.addWidget(QLabel("Action:"))
        opts.addWidget(self.action_cb, 1)

        self.mark_cb = QComboBox()
        self.mark_cb.addItems(
            [
                "auto",
                "gemini",
                "doubao",
                "dola",
                "jimeng",
                "qwen",
                "kling",
                "yuanbao",
                "samsung",
                "runninghub",
                "baidu",
                "liblib",
                "microsoft",
                "jimeng_pill",
            ]
        )
        # Mark only applies to removal actions that touch visible marks.
        self.mark_cb.setEnabled(False)
        opts.addWidget(QLabel("Mark:"))
        opts.addWidget(self.mark_cb, 1)
        opts.addStretch()

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25b6 Run")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(
            "padding: 8px 20px; border-radius: 4px; font-weight: bold;"
        )
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()

        self.log = LogPanel()
        layout.addWidget(self.dropzone)
        layout.addLayout(opts)
        layout.addLayout(btn_row)
        layout.addWidget(self.log, 1)

    def _on_files(self, paths: list[Path]) -> None:
        if paths:
            self._path = paths[0]
            self.run_btn.setEnabled(True)
            self.log.info(f"Loaded: {self._path.name}")

    def _on_action_changed(self, _action: str) -> None:
        """Enable the Mark selector only for actions that use visible marks."""
        uses_mark = self.action_cb.currentText() in ("Remove visible marks", "Remove all")
        self.mark_cb.setEnabled(uses_mark)

    def _run(self) -> None:
        if self._path is None:
            return
        action = self.action_cb.currentText()
        mark = self.mark_cb.currentText()
        self.log.info(f"Video {action} on {self._path.name}...")
        self.run_btn.setEnabled(False)

        src = self._path

        def _work() -> object:
            resolved_mark = mark
            if mark == "auto" and action in ("Remove visible marks", "Remove all"):
                from remove_ai_watermarks import identify_video

                identified = identify_video(src)
                if identified.visible_mark is not None:
                    resolved_mark = identified.visible_mark
            if action == "Identify":
                from remove_ai_watermarks import identify_video
                return identify_video(src)
            if action == "Remove visible marks":
                from remove_ai_watermarks import remove_video_visible
                return remove_video_visible(src, mark=resolved_mark)
            if action == "Remove metadata":
                from remove_ai_watermarks import remove_video_metadata
                return remove_video_metadata(src)
            # Remove all
            from remove_ai_watermarks import remove_video_all
            return remove_video_all(src, mark=resolved_mark)

        worker = Worker(_work)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self.log.error)
        worker.signals.finished.connect(lambda: self.run_btn.setEnabled(True))
        pool = QThreadPool.globalInstance()
        if pool is not None:
            pool.start(worker)

    def _on_result(self, result: object) -> None:
        if result is None:
            self.log.info("Done.")
        elif hasattr(result, "to_dict"):
            import json
            self.log.info(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        elif hasattr(result, "__dataclass_fields__"):
            import dataclasses
            self.log.info(str(dataclasses.asdict(result)))  # type: ignore[arg-type]
        else:
            self.log.info(str(result))


class MetadataTab(QWidget):
    """Metadata operations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._paths: list[Path] = []

        self.dropzone = DropZone(accept_multiple=True)
        self.dropzone.setMinimumHeight(80)
        self.dropzone.files_dropped.connect(self._on_files)

        opts = QHBoxLayout()
        self.action_cb = QComboBox()
        self.action_cb.addItems(
            ["Check for AI metadata", "Remove AI metadata", "Strip all metadata"]
        )
        opts.addWidget(QLabel("Action:"))
        opts.addWidget(self.action_cb, 1)

        self.keep_standard_cb = QCheckBox("Keep standard metadata")
        self.keep_standard_cb.setChecked(True)
        opts.addWidget(self.keep_standard_cb)
        opts.addStretch()

        self.run_btn = QPushButton("\u25b6 Run")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(
            "padding: 8px 20px; border-radius: 4px; font-weight: bold;"
        )

        self.log = LogPanel()
        layout.addWidget(self.dropzone)
        layout.addLayout(opts)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.log, 1)

    def _on_files(self, paths: list[Path]) -> None:
        self._paths = paths
        self.run_btn.setEnabled(len(paths) > 0)

    def _run(self) -> None:
        action = self.action_cb.currentText()
        self.run_btn.setEnabled(False)
        self.log.info(f"Metadata: {action} on {len(self._paths)} file(s)")

        paths = list(self._paths)
        keep_std = self.keep_standard_cb.isChecked()

        def _work() -> object:
            from remove_ai_watermarks.metadata import (
                get_ai_metadata,
                remove_ai_metadata,
                strip_and_verify,
            )

            results: dict[str, Any] = {}
            for p in paths:
                try:
                    if action == "Check for AI metadata":
                        md = get_ai_metadata(p)
                        results[p.name] = md or "No AI metadata detected"
                    elif action == "Remove AI metadata":
                        remove_ai_metadata(p, p)
                        results[p.name] = "Stripped"
                    else:
                        written, remaining = strip_and_verify(
                            p, p, keep_standard=keep_std
                        )
                        results[p.name] = f"Written={written}, remaining={remaining}"
                except Exception as e:
                    results[p.name] = f"Error: {e}"
            return results

        worker = Worker(_work)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self.log.error)
        worker.signals.finished.connect(lambda: self.run_btn.setEnabled(True))
        pool = QThreadPool.globalInstance()
        if pool is not None:
            pool.start(worker)

    def _on_result(self, result: object) -> None:
        if isinstance(result, dict):
            for name, status in result.items():
                self.log.info(f"{name}: {status}")
        self.run_btn.setEnabled(True)


# ── Main window ─────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Remove-AI-Watermarks")
        self.setMinimumSize(1000, 700)

        # Check dependencies
        if not CV2_AVAILABLE:
            QMessageBox.warning(
                self, "Missing Dependencies",
                "OpenCV (cv2) and NumPy are not installed.\n"
                "Run: uv sync --extra visible\n\n"
                "The GUI may be limited.",
            )

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(RemovalTab(), "\U0001f9f9 Remove Visible")
        tabs.addTab(FullCleanTab(), "\U0001f52c Full Clean")
        tabs.addTab(BatchTab(), "\U0001f4c1 Batch")
        tabs.addTab(IdentifyTab(), "\U0001f50d Identify")
        tabs.addTab(EraseTab(), "\u2702\ufe0f Erase")
        tabs.addTab(VideoTab(), "\U0001f3ac Video")
        tabs.addTab(MetadataTab(), "\U0001f3f7\ufe0f Metadata")

        self.setCentralWidget(tabs)

        # Menu
        self._build_menu()

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        if menubar is None:
            return
        file_menu = menubar.addMenu("&File")
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menubar.addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _show_about(self) -> None:
        from remove_ai_watermarks import __version__

        QMessageBox.about(
            self, "About Remove-AI-Watermarks",
            f"<b>Remove-AI-Watermarks v{__version__}</b><br><br>"
            + "Remove visible and invisible AI watermarks "
            "from images and video.<br><br>"
            "See: <a href='https://github.com/wiltodelta/"
            "remove-ai-watermarks'>GitHub</a>",
        )


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Remove-AI-Watermarks")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
