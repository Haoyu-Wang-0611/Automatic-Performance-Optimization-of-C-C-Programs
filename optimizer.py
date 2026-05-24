import subprocess
import re
import os
import sys
import shutil
import time
import random
import argparse
import json
import csv
from pathlib import Path
from enum import Enum
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 配置API接口和模型
# ---------------------------------------------------------------------------
DEFAULT_OPENAI_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

# DEFAULT_MODEL = "Pro/zai-org/GLM-5"
# DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
# DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1"
# DEFAULT_MODEL = "Pro/MiniMaxAI/MiniMax-M2.5"
# DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"
# DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3"
ROOT_DIR = Path(__file__).parent.resolve()
BENCHMARKS_FILE = ROOT_DIR / "benchmarks.json"
WORK_ROOT = ROOT_DIR / "work"
VERIFICATION_SCRIPT = ROOT_DIR / "accuracy verification.py"
RESULTS_DIR = ROOT_DIR / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-driven C/C++ performance optimizer."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL,
        help="Model name. Defaults to LLM_MODEL/OPENAI_MODEL or the built-in default.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
        help="API base URL. Defaults to OPENAI_BASE_URL or the built-in default.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--benchmark",
        default="all",
        help="Benchmark name from benchmarks.json, or 'all'.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 全局优化参数配置
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 3        # 允许的最大优化迭代次数
MAX_COMPILE_RETRIES = 2   # 允许的编译修复尝试次数
BASELINE_REFERENCE_RUNS = 5  # 固定基准时间的采样次数（取中位数）


# ---------------------------------------------------------------------------
# 终端交互分隔线配置（统一长度）
# ---------------------------------------------------------------------------
UI_LINE_WIDTH = 70
UI_MAJOR_LINE = "=" * UI_LINE_WIDTH
UI_MINOR_LINE = "─" * UI_LINE_WIDTH


# ---------------------------------------------------------------------------
# 系统级Prompt：定义LLM的角色和优化规则
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert C/C++ performance optimization engineer.
Your goal is to improve runtime performance while preserving semantics exactly.

Priorities:
1. Preserve correctness exactly.
2. Preserve the full program I/O protocol exactly.
3. Prefer source-to-source transformations in standard C/C++.
4. Use compiler diagnostics and runtime profiling together.
5. Improve vectorization opportunities, memory locality, and branch behavior.
6. Do not remove validation logic.
7. Do not change output file names, output row counts, output column counts, or stdout structure.

Critical constraints:
- Output protocol is part of semantics(The semantics of the optimized code must be the same as that of the original code).
- Do not subsample, compress, truncate, summarize, or reduce final outputs.
- If the baseline writes a file, the optimized version must write the same file with the same number of numeric values and the same textual structure.
- Keep the code as close to the original as possible when uncertain.

Return only the full optimized source code inside one ```cpp code block.
"""


# ---------------------------------------------------------------------------
# 构建一个发送给LLM的Prompt，要求AI根据提供的性能分析数据来优化一段C/C++代码
# ---------------------------------------------------------------------------
OPT_PROMPT_TEMPLATE = """\
Optimize the following C/C++ program using both runtime profiling and compiler diagnostics.

## Runtime Profiling
- Cycles: {cycles}
- Instructions: {instructions}
- L1-dcache-load-misses: {l1_misses}
- Branch-misses: {branch_misses}
- Wall time: {elapsed} seconds

## Hotspots
{hotspot_text}

## Compiler Vectorization Diagnostics
{compiler_feedback}

## Automatically Derived Facts
{derived_facts}

## Current Source Code
```cpp
{source_code}
```

## Optimization Requirements
- Preserve exact semantics.
- Keep validation logic intact.
- Preserve the full I/O protocol exactly.
- Do not change stdout message structure.
- Do not change output filenames.
- Do not change output row count or column count.
- Do not subsample, truncate, compress, or summarize file output.
- Prefer source-level transformations that help auto-vectorization.
- Consider loop splitting, loop reordering, temporary variables, branch simplification, scalar replacement, and memory locality improvements.
- Do not use platform-specific intrinsics unless explicitly requested.

Return the complete optimized source code only inside one ```cpp code block.
"""
###-------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 错误修复Prompt模板：编译或验证失败时的反馈机制
# ---------------------------------------------------------------------------
ERROR_PROMPT_TEMPLATE = """\
The previous optimization attempt failed.

Error / Feedback:
{error}

Here is the current source code:

```cpp
{source_code}
```

Fix the issue and return the complete, corrected C++ source code.

## Critical constraints:
- The output protocol must exactly match the baseline.
- Do not change stdout structure.
- Do not change output filenames.
- Do not change output row count or output column count.
- Do not subsample, truncate, compress, or summarize output.
"""


# ---------------------------------------------------------------------------
# 状态枚举
# ---------------------------------------------------------------------------
class Status(Enum):
    SUCCESS = "Success"
    COMPILATION_ERROR = "Compilation Error"
    VALIDATION_FAILED = "Validation Failed"
    RUNTIME_ERROR = "Runtime Error"


# ---------------------------------------------------------------------------
# 编译器反馈数据结构：解析Clang优化报告
# ---------------------------------------------------------------------------
@dataclass
class CompilerLoopRemark:
    kind: str                 # "vectorized" / "missed" / "analysis"
    line: Optional[int]
    column: Optional[int]
    message: str

@dataclass
class CompilerFeedback:
    remarks: list[CompilerLoopRemark] = field(default_factory=list)
    raw_output: str = ""

    def summary_text(self, max_items: int = 20) -> str:
        if not self.remarks:
            return "  (no compiler vectorization feedback available)"

        seen = set()
        lines = []
        for r in self.remarks:
            key = (r.kind, r.line, r.column, r.message)
            if key in seen:
                continue
            seen.add(key)

            loc = f"line {r.line}:{r.column}" if r.line is not None else "unknown location"
            lines.append(f"- [{r.kind}] {loc}: {r.message}")

            if len(lines) >= max_items:
                break

        return "\n".join(lines)


@dataclass
class ProgramAnalysis:
    summary_lines: list[str] = field(default_factory=list)

    def prompt_text(self) -> str:
        lines: list[str] = []
        if self.summary_lines:
            lines.append("Observed Facts:")
            lines.extend(f"- {line}" for line in self.summary_lines)
        return "\n".join(lines) if lines else "  (no derived facts available)"


@dataclass
class BenchmarkConfig:
    name: str
    baseline_source: Path
    output_files: list[str] = field(default_factory=list)
    time_pattern: str = r"completed in\s+([\d.eE+-]+)\s+seconds"

    @property
    def work_dir(self) -> Path:
        return WORK_ROOT / self.name

    @property
    def target_src(self) -> Path:
        return self.work_dir / "target.cpp"

    @property
    def target_orig(self) -> Path:
        return self.work_dir / "target_baseline.cpp"

    @property
    def target_bin(self) -> Path:
        return self.work_dir / "target_bin"

    @property
    def baseline_bin(self) -> Path:
        return self.work_dir / "baseline_bin"

    @property
    def backup_dir(self) -> Path:
        return self.work_dir / "backups"

    @property
    def perf_data(self) -> Path:
        return self.work_dir / "perf.data"

    @property
    def analysis_bin(self) -> Path:
        return self.work_dir / "analysis_bin"


def load_benchmark_configs(selected_name: str) -> list[BenchmarkConfig]:
    if not BENCHMARKS_FILE.exists():
        raise FileNotFoundError(f"Benchmark config not found: {BENCHMARKS_FILE}")

    raw_configs = json.loads(BENCHMARKS_FILE.read_text())
    configs: list[BenchmarkConfig] = []
    for item in raw_configs:
        configs.append(
            BenchmarkConfig(
                name=item["name"],
                baseline_source=(ROOT_DIR / item["baseline_source"]).resolve(),
                output_files=item.get("output_files", []),
                time_pattern=item.get(
                    "time_pattern",
                    r"completed in\s+([\d.eE+-]+)\s+seconds",
                ),
            )
        )

    if selected_name == "all":
        return configs

    for config in configs:
        if config.name == selected_name:
            return [config]

    available = ", ".join(config.name for config in configs)
    raise ValueError(
        f"Unknown benchmark '{selected_name}'. Available benchmarks: {available}"
    )


def append_summary_row(row: dict[str, str | int | float]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "benchmark",
        "model",
        "baseline_time",
        "best_opt_time",
        "speedup",
        "verification_status",
        "iterations_used",
    ]
    write_header = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# 优化器核心类：管理编译、运行、分析和LLM交互的生命周期
# ---------------------------------------------------------------------------
class Optimizer:
    def __init__(self, benchmark: BenchmarkConfig, model: str, base_url: str, api_key: str):
        if not api_key:
            raise ValueError(
                "Missing API key. Set OPENAI_API_KEY or pass --api-key."
            )

        self.benchmark = benchmark
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.conversation_history: list[dict] = []
        self.iteration = 0

    @staticmethod
    def _compress_user_content(content: str) -> str:
        content = re.sub(
            r"(## Current Source Code\n```cpp\n).*?(\n```)",
            r"\1// [Source code omitted from history to save context]\2",
            content,
            flags=re.DOTALL,
        )
        content = re.sub(
            r"(Here is the current source code:\n\n```cpp\n).*?(\n```)",
            r"\1// [Source code omitted from history to save context]\2",
            content,
            flags=re.DOTALL,
        )
        return content

    @staticmethod
    def _compress_assistant_content(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("#include"):
            return "```cpp\n// [Optimization code emitted in previous turn omitted]\n```"

        return re.sub(
            r"```(?:c|cpp|c\+\+)?\s*\n(.*?)```",
            "```cpp\n// [Optimization code emitted in previous turn omitted]\n```",
            content,
            flags=re.DOTALL,
        )

    def build_messages(self, user_msg: str) -> list[dict]:
        compressed_history: list[dict] = []
        for msg in self.conversation_history:
            content = msg["content"]
            if msg["role"] == "user":
                content = self._compress_user_content(content)
            elif msg["role"] == "assistant":
                content = self._compress_assistant_content(content)
            compressed_history.append({
                "role": msg["role"],
                "content": content,
            })

        return [{"role": "system", "content": SYSTEM_PROMPT}] + compressed_history + [
            {"role": "user", "content": user_msg}
        ]

    def prepare_workspace(self) -> None:
        self.benchmark.work_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark.backup_dir.mkdir(exist_ok=True)
        shutil.copy2(self.benchmark.baseline_source, self.benchmark.target_orig)
        shutil.copy2(self.benchmark.baseline_source, self.benchmark.target_src)


    # ---------------------------------------------------------------------------
    # [Module 1] 编译模块：负责调用g++编译源代码
    # ---------------------------------------------------------------------------
    def compile_target(self, source_file: Path, output_file: Path) -> tuple[Status, str]:
        """Compile source with g++ -O3 -g. Returns (status, error_message)."""
        cmd = ["g++", "-O3", "-g", "-o", str(output_file), str(source_file)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.benchmark.work_dir,
        )
        if result.returncode != 0:
            return Status.COMPILATION_ERROR, result.stderr
        return Status.SUCCESS, ""


    # ---------------------------------------------------------------------------
    # [Module 2] 执行与验证模块：运行二进制并进行性能与正确性校验
    # ---------------------------------------------------------------------------
    def run_target(self, executable: Path, seed: int = None) -> tuple[Status, float, str]:
        """Run binary (no arguments). Parse time from stdout."""
        cmd = [str(executable)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.benchmark.work_dir,
            )
        except subprocess.TimeoutExpired:
            return Status.RUNTIME_ERROR, 0.0, "Timeout"

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        if result.returncode != 0:
             detail = stdout + stderr if (stdout or stderr) else f"exit code {result.returncode}"
             return Status.RUNTIME_ERROR, 0.0, detail

        # Parse Time from STDOUT (Format: "Matrix multiplication completed in 1.234 seconds.")
        # We accept float with optional scientific notation just in case
        m = re.search(self.benchmark.time_pattern, stdout)
        elapsed = float(m.group(1)) if m else 0.0
        
        # If we didn't find the time message, something is wrong
        if elapsed == 0.0 and "completed in" not in stdout:
             return Status.RUNTIME_ERROR, 0.0, f"Output format error: {stdout}"

        return Status.SUCCESS, elapsed, stdout


    # ---------------------------------------------------------------------------
    # [Module 3] 验证脚本接口：调用外部 accuracy verification.py 进行深度校验
    # ---------------------------------------------------------------------------
    def run_external_verification(self) -> tuple[bool, str]:
        """Run the external accuracy verification script."""
        verify_script = VERIFICATION_SCRIPT

        if not verify_script.exists():
            return False, f"Verification script not found: {verify_script}"

        try:
            res = subprocess.run(
                [sys.executable, str(verify_script)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.benchmark.work_dir,
                env={
                    **os.environ,
                    "OPTIMIZER_WORK_DIR": str(self.benchmark.work_dir),
                },
            )
        except subprocess.TimeoutExpired:
            return False, "Verification timeout"

        output = (res.stdout or "") + (res.stderr or "")
        return res.returncode == 0, output


    # ---------------------------------------------------------------------------
    # [Module 4] 性能分析模块：包含 perf stat 和 perf record/report
    # ---------------------------------------------------------------------------
    def run_profiler(self, executable: Path) -> dict:
        """Run perf stat + perf record/report. Returns metrics dict."""
        metrics: dict = {}

        # --- perf stat ---
        stat_cmd = [
            "perf", "stat", "-e",
            "cycles,instructions,L1-dcache-load-misses,branch-misses",
            str(executable),
        ]
        try:
            stat_result = subprocess.run(
                stat_cmd, capture_output=True, text=True, timeout=120
            )
            stat_text = stat_result.stderr  # perf stat outputs to stderr

            event_re = re.compile(
                r"^\s*([\d,]+)\s+(cycles|instructions|L1-dcache-load-misses|branch-misses)",
                re.MULTILINE,
            )
            for m in event_re.finditer(stat_text):
                val = int(m.group(1).replace(",", ""))
                metrics[m.group(2)] = val

            time_m = re.search(r"([\d.]+)\s+seconds time elapsed", stat_text)
            if time_m:
                metrics["elapsed_seconds"] = float(time_m.group(1))
        except Exception as e:
            print(f"  WARNING: perf stat failed: {e}")

        # --- perf record + report ---
        hotspots: list[dict] = []
        report_raw = ""
        try:
            subprocess.run(
                ["perf", "record", "-F", "99", "-g",
                 "-o", str(self.benchmark.perf_data), "--", str(executable)],
                capture_output=True, text=True, timeout=120, cwd=self.benchmark.work_dir,
            )
            report_result = subprocess.run(
                ["perf", "report", "--stdio", "--no-children",
                 "--sort=dso,sym,srcline", "-i", str(self.benchmark.perf_data)],
                capture_output=True, text=True, timeout=30, cwd=self.benchmark.work_dir,
            )
            report_raw = report_result.stdout

            # Format: "  85.23%  target_bin  [.] matmul()  target.cpp:12"
            hotspot_re = re.compile(
                r"^\s*([\d.]+)%\s+\S+\s+\[.\]\s+(.+?)\s{2,}(\S+:\d+)",
                re.MULTILINE,
            )
            for m in hotspot_re.finditer(report_raw):
                pct = float(m.group(1))
                if pct >= 1.0:
                    hotspots.append({
                        "percentage": pct,
                        "symbol": m.group(2).strip(),
                        "source_line": m.group(3),
                    })
        except Exception as e:
            print(f"  WARNING: perf record/report failed: {e}")

        metrics["hotspots"] = hotspots
        metrics["perf_report_raw"] = report_raw[:3000]
        return metrics


    # ---------------------------------------------------------------------------
    # [Module 5] 编译器分析模块：使用Clang Remarks读取向量化反馈
    # ---------------------------------------------------------------------------
    def run_compiler_analysis(self, source_file: Path) -> CompilerFeedback:
        """
        Use Clang vectorization remarks to collect loop-level compiler feedback.
        """
        feedback = CompilerFeedback()

        cmd = [
            "clang++",
            "-O3",
            "-g",
            "-Rpass=loop-vectorize",
            "-Rpass-missed=loop-vectorize",
            "-Rpass-analysis=loop-vectorize",
            "-fno-color-diagnostics",
            str(source_file),
            "-o",
            str(self.benchmark.analysis_bin),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            text = (result.stdout or "") + "\n" + (result.stderr or "")
            feedback.raw_output = text

            remark_re = re.compile(
                r"^(.*?):(\d+):(\d+):\s+remark:\s+(.*?)\s+\[-Rpass(?:-missed|-analysis)?=loop-vectorize\]",
                re.MULTILINE,
            )

            for m in remark_re.finditer(text):
                line = int(m.group(2))
                col = int(m.group(3))
                msg = m.group(4).strip()

                lower_msg = msg.lower()
                if "not vectorized" in lower_msg:
                    kind = "missed"
                elif "vectorized loop" in lower_msg or lower_msg.startswith("vectorized"):
                    kind = "vectorized"
                else:
                    kind = "analysis"

                feedback.remarks.append(
                    CompilerLoopRemark(
                        kind=kind,
                        line=line,
                        column=col,
                        message=msg,
                    )
                )

        except Exception as e:
            print(f"  WARNING: compiler analysis failed: {e}")

        return feedback

    def analyze_program_features(
        self,
        source_code: str,
        profile_data: dict,
        compiler_feedback: CompilerFeedback,
    ) -> ProgramAnalysis:
        analysis = ProgramAnalysis()

        lines = source_code.splitlines()
        loop_lines = [idx + 1 for idx, line in enumerate(lines) if re.search(r"\bfor\s*\(", line)]
        if loop_lines:
            analysis.summary_lines.append(
                f"Detected {len(loop_lines)} for-loops; first loop lines: {', '.join(map(str, loop_lines[:6]))}"
            )

        if re.search(r"\bfor\s*\([^)]*\)\s*\{?(?:\s*//.*)?\s*\n(?:.*\n){0,6}.*\bfor\s*\(", source_code, re.DOTALL):
            analysis.summary_lines.append(
                "Contains nested loop regions, so loop ordering and locality are likely important."
            )

        array_accesses = re.findall(r"\b([A-Za-z_]\w*)\s*\[[^\]]+\]\s*\[[^\]]+\]", source_code)
        if array_accesses:
            unique_arrays = sorted(set(array_accesses))
            preview = ", ".join(unique_arrays[:6])
            analysis.summary_lines.append(
                f"Uses multi-dimensional array-style accesses on: {preview}"
            )

        math_calls = re.findall(r"\b(exp|log|pow|sqrt|sin|cos)\s*\(", source_code)
        if math_calls:
            unique_math = sorted(set(math_calls))
            analysis.summary_lines.append(f"Contains expensive math calls: {', '.join(unique_math)}")

        loop_blocks = re.findall(r"\bfor\s*\([^)]*\)\s*\{.*?\}", source_code, re.DOTALL)
        loop_text = "\n".join(loop_blocks[:12])

        if re.search(r"\b(void|int|float|double|char)\s*\*\s*", source_code):
            analysis.summary_lines.append("Detected pointer variables/parameters.")

        if re.search(r"(?<!/)/(?!/)", loop_text):
            analysis.summary_lines.append("Detected division operations inside loop bodies.")

        if re.search(r"\bif\s*\(", loop_text):
            analysis.summary_lines.append("Detected conditional branches inside loop bodies.")

        if "fprintf" in source_code or "fopen" in source_code or "printf" in source_code:
            analysis.summary_lines.append(
                "Contains explicit output formatting, so I/O protocol preservation is mandatory."
            )

        if re.search(r"\b(sum|acc|accum|total)\b", source_code):
            analysis.summary_lines.append(
                "Contains accumulator-style variables, which suggests reduction-like loop bodies."
            )

        hotspots = profile_data.get("hotspots", [])
        if hotspots:
            top = hotspots[:3]
            hotspot_desc = ", ".join(
                f"{h['symbol']} ({h['percentage']:.1f}%)" for h in top
            )
            analysis.summary_lines.append(f"Top hotspots: {hotspot_desc}")

        if profile_data.get("L1-dcache-load-misses", 0):
            analysis.summary_lines.append(
                f"L1-dcache-load-misses: {profile_data.get('L1-dcache-load-misses', 0):,}"
            )

        missed_msgs = [r.message for r in compiler_feedback.remarks if r.kind == "missed"]
        if missed_msgs:
            analysis.summary_lines.append(
                f"Compiler reported {len(missed_msgs)} missed vectorization opportunities."
            )
            joined = " ".join(missed_msgs).lower()
            if "cannot prove it is safe to reorder floating-point operations" in joined:
                analysis.summary_lines.append(
                    "Compiler reported floating-point reordering legality concerns."
                )
            if "call instruction cannot be vectorized" in joined:
                analysis.summary_lines.append(
                    "Compiler reported call instructions blocking vectorization."
                )
            if "not vectorized" in joined:
                analysis.summary_lines.append(
                    "Compiler reported non-vectorized loop regions."
                )
            if "unsafe dependent memory operations" in joined or "dependence" in joined:
                analysis.summary_lines.append(
                    "Compiler reported dependency-related vectorization blockers."
                )

        vectorized_count = sum(1 for r in compiler_feedback.remarks if r.kind == "vectorized")
        if vectorized_count:
            analysis.summary_lines.append(
                f"Compiler already vectorized {vectorized_count} loop(s); preserve those profitable loop shapes."
            )

        deduped_summary = list(dict.fromkeys(analysis.summary_lines))
        analysis.summary_lines = deduped_summary[:8]
        return analysis


    # ---------------------------------------------------------------------------
    # [Module 6] LLM 交互模块：组装上下文并发送请求
    # ---------------------------------------------------------------------------
    def query_llm_for_optimization(
        self,
        source_code: str,
        profile_data: dict,
        program_analysis: ProgramAnalysis | None = None,
        compiler_feedback: str | None = None,
        error_context: str | None = None
    ) -> str:
        """Send source + profiling data + compiler feedback to LLM, return raw response text."""

        if compiler_feedback is None:
            compiler_feedback = "  (no compiler feedback available)"
        if program_analysis is None:
            derived_facts = "  (no derived facts available)"
        else:
            derived_facts = program_analysis.prompt_text()

        if error_context:
            user_msg = ERROR_PROMPT_TEMPLATE.format(
                error=error_context, source_code=source_code
            )
        else:
            hotspot_lines = ""
            for h in profile_data.get("hotspots", []):
                hotspot_lines += (
                    f"  {h['percentage']:5.1f}%  {h['symbol']}  {h['source_line']}\n"
                )
            if not hotspot_lines:
                hotspot_lines = "  (no hotspot data available)\n"

            user_msg = OPT_PROMPT_TEMPLATE.format(
                cycles=f"{profile_data.get('cycles', 0):,}",
                instructions=f"{profile_data.get('instructions', 0):,}",
                l1_misses=f"{profile_data.get('L1-dcache-load-misses', 0):,}",
                branch_misses=f"{profile_data.get('branch-misses', 0):,}",
                elapsed=profile_data.get("elapsed_seconds", "N/A"),
                hotspot_text=hotspot_lines,
                compiler_feedback=compiler_feedback,
                derived_facts=derived_facts,
                source_code=source_code,
            )

        messages = self.build_messages(user_msg)

        print(UI_MAJOR_LINE)
        print(" Compiler Feedback Passed to LLM ")
        print(compiler_feedback)
        print(UI_MAJOR_LINE)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        reply = response.choices[0].message.content
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply


    # ---------------------------------------------------------------------------
    # [Module 7] 代码提取与应用模块：处理Markdown块
    # ---------------------------------------------------------------------------
    @staticmethod
    def extract_code(response: str) -> str | None:
        """Extract C/C++ code from markdown code block."""
        m = re.search(r"```(?:c|cpp|c\+\+)?\s*\n(.*?)```", response, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: raw code starting with #include
        if response.strip().startswith("#include"):
            return response.strip()
        return None


    # ---------------------------------------------------------------------------
    # [Module 8] 优化应用模块：文件写入与备份管理
    # ---------------------------------------------------------------------------
    def apply_optimization(self, code: str) -> bool:
        """Backup original file and write new code."""
        self.benchmark.backup_dir.mkdir(exist_ok=True)
        backup = self.benchmark.backup_dir / f"target_iter{self.iteration}.cpp"
        shutil.copy2(self.benchmark.target_src, backup)
        self.benchmark.target_src.write_text(code)
        return True


    # ---------------------------------------------------------------------------
    # [Module 9] 核心主循环：迭代优化控制逻辑
    # ---------------------------------------------------------------------------
    def main_loop(self):
        self.prepare_workspace()
        baseline_time = 0.0
        best_opt_time = 0.0
        best_speedup_so_far = 0.0
        verify_passed = False
        iterations_used = 0
        print("\n" + UI_MAJOR_LINE)
        print(" Mini-PerfLLM: LLM-Driven C/C++ Performance Optimization")
        print(UI_MAJOR_LINE)
        print(f" Benchmark name : {self.benchmark.name}")
        print(f" Work directory : {self.benchmark.work_dir}")
        # Always reset target.cpp from immutable baseline
        shutil.copy2(self.benchmark.target_orig, self.benchmark.target_src)
        # ---- Phase 1: Baseline ----
        print("\n" + UI_MAJOR_LINE)
        print("[Phase 1] Baseline Profiling")
        print(UI_MAJOR_LINE)
        shutil.copy2(self.benchmark.target_orig, self.benchmark.target_src)
        
        # Compile baseline to separate binary (BASELINE_BIN)
        status, err = self.compile_target(self.benchmark.target_src, self.benchmark.baseline_bin)
        if status != Status.SUCCESS:
            print(f"FATAL: Baseline compilation failed:\n{err}")
            append_summary_row({
                "benchmark": self.benchmark.name,
                "model": self.model,
                "baseline_time": "",
                "best_opt_time": "",
                "speedup": "",
                "verification_status": "verification_not_passed",
                "iterations_used": iterations_used,
            })
            return
            
        print("[Init] Baseline compiled successfully.")

        # ---- Phase 1: Baseline Metrics ----
        print("\n[Phase 1] Profiling Baseline (Fixed Input)...")
        print(UI_MINOR_LINE)

        # Run baseline multiple times, then fix one reference baseline time (median)
        baseline_samples: list[float] = []
        for _ in range(BASELINE_REFERENCE_RUNS):
            status, sample_time, _ = self.run_target(self.benchmark.baseline_bin)
            if status != Status.SUCCESS:
                print("Baseline Runtime Error during profiling!")
                append_summary_row({
                    "benchmark": self.benchmark.name,
                    "model": self.model,
                    "baseline_time": "",
                    "best_opt_time": "",
                    "speedup": "",
                    "verification_status": "verification_not_passed",
                    "iterations_used": iterations_used,
                })
                return
            baseline_samples.append(sample_time)

        baseline_samples.sort()
        baseline_time = baseline_samples[len(baseline_samples) // 2]
        if baseline_time <= 0:
            baseline_time = 0.001

        print(f"  Execution time (fixed median of {BASELINE_REFERENCE_RUNS} runs) : {baseline_time:.3f}s")
        
        baseline_metrics = self.run_profiler(self.benchmark.baseline_bin)
        bl_cycles = baseline_metrics.get("cycles", 0)
        bl_misses = baseline_metrics.get("L1-dcache-load-misses", 0)

        # Initial Compiler Feedback
        baseline_compiler_feedback = self.run_compiler_analysis(self.benchmark.target_src)
        print("  Compiler feedback:")
        print(baseline_compiler_feedback.summary_text())

        print(f"  Cycles         : {bl_cycles:,}")
        print(f"  L1-dcache-miss : {bl_misses:,}")

        original_source = self.benchmark.target_src.read_text()
        last_working_source = original_source
        last_working_metrics = dict(baseline_metrics)
        best_speedup_so_far = 1.0
        best_opt_time = baseline_time

        # ---- Phase 2: Iterative Optimization ----
        for iteration in range(1, MAX_ITERATIONS + 1):
            self.iteration = iteration
            iterations_used = iteration
            print("\n" + UI_MINOR_LINE)
            print(f"[Phase 2] Iteration {iteration}/{MAX_ITERATIONS}")
            print(UI_MINOR_LINE)

            current_source = self.benchmark.target_src.read_text()
            
            # If iteration > 1, we profile TARGET_BIN (the optimized binary), else baseline
            current_metrics = (
                self.run_profiler(self.benchmark.target_bin) if iteration > 1 else baseline_metrics
            )

            # Query LLM
            print("  Querying LLM for optimization...")
            compiler_output = self.run_compiler_analysis(self.benchmark.target_src)
            program_analysis = self.analyze_program_features(
                current_source,
                current_metrics,
                compiler_output,
            )
            try:
                llm_response = self.query_llm_for_optimization(
                    current_source,
                    current_metrics,
                    program_analysis,
                    compiler_output.summary_text()
                )
            except Exception as e:
                print(f"  ERROR: LLM API call failed: {e}")
                continue

            new_code = self.extract_code(llm_response)
            if new_code is None:
                print("  WARNING: Could not extract code from LLM response. Skipping.")
                continue

            # Apply and compile (with retries)
            self.apply_optimization(new_code)
            
            compiled = False
            for retry in range(MAX_COMPILE_RETRIES + 1):
                # Compile to TARGET_BIN
                status, err = self.compile_target(self.benchmark.target_src, self.benchmark.target_bin)
                if status == Status.SUCCESS:
                    compiled = True
                    break
                print(f"  Compilation failed (attempt {retry + 1}). Asking LLM to fix...")
                try:
                    repair_compiler_output = self.run_compiler_analysis(self.benchmark.target_src)
                    repair_analysis = self.analyze_program_features(
                        self.benchmark.target_src.read_text(),
                        current_metrics,
                        repair_compiler_output,
                    )
                    llm_response = self.query_llm_for_optimization(
                        self.benchmark.target_src.read_text(),
                        current_metrics,
                        repair_analysis,
                        repair_compiler_output.summary_text(),
                        error_context=err
                    )
                except Exception as e:
                    print(f"  ERROR: LLM API call failed: {e}")
                    break
                
                fixed = self.extract_code(llm_response)
                if fixed:
                    self.benchmark.target_src.write_text(fixed)
                else:
                    print("  WARNING: Could not extract fix from LLM response.")
                    break

            if not compiled:
                print("  Compilation failed after retries. Reverting.")
                self.benchmark.target_src.write_text(last_working_source)
                self.compile_target(self.benchmark.target_src, self.benchmark.target_bin)
                continue

            # # --- VALIDATION STEP ---
            # # Compare output of TARGET_BIN vs BASELINE_BIN
            # # NOTE: Without validation output, this only checks if it runs successfully
            # passed, err_msg, opt_time = self.validate_equivalence(TARGET_BIN, BASELINE_BIN)
            
            # if not passed:
            #     print(f"  Validation Failed: {err_msg}")
            #     print("  Reverting to last working version.")
            #     TARGET_SRC.write_text(last_working_source)
            #     self.compile_target(TARGET_SRC, TARGET_BIN)
            #     continue

            # # Re-measure baseline time for fairness
            # _, bl_time_current, _ = self.run_target(BASELINE_BIN)
            # if bl_time_current == 0: bl_time_current = 0.001

            # print(f"  Current Run: Opt={opt_time:.4f}s vs Base={bl_time_current:.4f}s")
            # 代码修改3
            # --- STRONG VALIDATION STEP ---
            verify_ok, verify_msg = self.run_external_verification()
            if not verify_ok:
                print("  External verification failed. Reverting to last working version.")
                # print(verify_msg[:1200])
                print("[DETAIL] Verification output:")
                print(UI_MAJOR_LINE)
                print(verify_msg[:600])
                print(UI_MAJOR_LINE)
                self.benchmark.target_src.write_text(last_working_source)
                self.compile_target(self.benchmark.target_src, self.benchmark.target_bin)
                continue

            # Only after correctness passes do we measure time
            status_opt, opt_time, output_opt = self.run_target(self.benchmark.target_bin)
            if status_opt != Status.SUCCESS:
                print(f"  Optimized run failed: {output_opt}")
                self.benchmark.target_src.write_text(last_working_source)
                self.compile_target(self.benchmark.target_src, self.benchmark.target_bin)
                continue

            # Use fixed baseline reference from Phase 1
            bl_time_current = baseline_time

            print(f"  Current Run: Opt={opt_time:.4f}s vs Base={bl_time_current:.4f}s")


            current_speedup = bl_time_current / opt_time if opt_time > 0 else 0
            
            # Profile optimized version
            opt_metrics = self.run_profiler(self.benchmark.target_bin)
            
            # --- Print newly optimized metrics so user can see them ---
            opt_cycles = opt_metrics.get("cycles", 0)
            opt_misses = opt_metrics.get("L1-dcache-load-misses", 0)
            opt_branch_misses = opt_metrics.get("branch-misses", 0)
            print(f"  [New Metrics] Cycles: {opt_cycles:,} | L1-Miss: {opt_misses:,} | Branch-Miss: {opt_branch_misses:,}")

            # Use strict improvement criteria to avoid accepting noisy regressions
            if current_speedup > best_speedup_so_far:
                print(f"  Accepted Verified (New Best Speedup: {current_speedup:.4f}x)")
                last_working_source = self.benchmark.target_src.read_text()
                last_working_metrics = opt_metrics
                best_speedup_so_far = current_speedup
                best_opt_time = opt_time
            else:
                print(f"  Reverting. Speedup {current_speedup:.4f}x is not better than best so far ({best_speedup_so_far:.4f}x).")
                self.benchmark.target_src.write_text(last_working_source)
                self.compile_target(self.benchmark.target_src, self.benchmark.target_bin)

        # ---- Phase 3: Final Verification & Report ----
        print("\n" + UI_MAJOR_LINE)
        print(" [Phase 3] FINAL VERIFICATION & REPORT")
        print(UI_MAJOR_LINE)

        VERIFY_SCRIPT = VERIFICATION_SCRIPT
        MAX_REPAIRS = 5
        repair_attempts = 0

        while repair_attempts <= MAX_REPAIRS:
            # 1. Run external verification script
            print(f"  [Verification] Running 'accuracy verification.py' (Attempt {repair_attempts})...")
            
            if not VERIFY_SCRIPT.exists():
                print(f"  [ERROR] Script not found: {VERIFY_SCRIPT}")
                break

            verify_cmd = [sys.executable, str(VERIFY_SCRIPT)]
            try:
                # Capture output to feed back to LLM if needed
                verify_res = subprocess.run(
                    verify_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=self.benchmark.work_dir,
                    env={
                        **os.environ,
                        "OPTIMIZER_WORK_DIR": str(self.benchmark.work_dir),
                    },
                )
            except subprocess.TimeoutExpired:
                print("  [ERROR] Verification script timed out.")
                verify_res = None

            # 2. Check Result
            if verify_res and verify_res.returncode == 0:
                print("  [SUCCESS] Verification Passed!")
                if verify_res.stdout:
                    print(verify_res.stdout)
                verify_passed = True
                break
            
            # 3. Handle Failure
            if repair_attempts >= MAX_REPAIRS:
                print("  [FAILURE] Max repair attempts reached. Optimization may be incorrect.")
                if verify_res:
                    print(f"  Last Output:\n{verify_res.stdout}")
                break
            
            print(f"  [FAILURE] Verification Failed! Asking LLM to fix...")
            # Print partial output for user visibility
            if verify_res:
                print(f"  Output:\n{verify_res.stdout[:1000]}") # Show first 1000 chars

            # 4. Prepare Context for Repair
            # We assume the current target source is the one that failed.
            current_source = self.benchmark.target_src.read_text()
            
            # Profile the likely-broken code to keep performance context (optional but good)
            current_metrics = self.run_profiler(self.benchmark.target_bin) 
            compiler_out = self.run_compiler_analysis(self.benchmark.target_src)
            
            # Construct feedback message
            error_msg = (
                f"The code failed external verification (correctness check).\n"
                f"Output from verification:\n"
                f"{verify_res.stdout if verify_res else 'Timeout/Error'}\n\n"
                f"Please fix the logic to ensure it matches the baseline exactly, "
                f"while maintaining performance."
            )

            # 5. Query LLM
            try:
                print("  Querying LLM for repair...")
                repair_analysis = self.analyze_program_features(
                    current_source,
                    current_metrics,
                    compiler_out,
                )
                llm_response = self.query_llm_for_optimization(
                    current_source,
                    current_metrics,
                    repair_analysis,
                    compiler_out.summary_text(),
                    error_context=error_msg
                )
            except Exception as e:
                print(f"  [ERROR] LLM API call failed: {e}")
                break

            # 6. Apply Fix
            fixed_code = self.extract_code(llm_response)
            if not fixed_code:
                print("  [WARNING] Could not extract fix from LLM response.")
                break
            
            self.apply_optimization(fixed_code)
            
            # 7. Compile New Candidate
            status, err = self.compile_target(self.benchmark.target_src, self.benchmark.target_bin)
            if status != Status.SUCCESS:
                print(f"  [Repair] Compilation failed: {err}")
                
            repair_attempts += 1


        if not verify_passed:
            print("\n  [CRITICAL] Final Verification Failed after repairs.")
            print("  Reverting to last known working version (Baseline or safe Optimization).")
            self.benchmark.target_src.write_text(last_working_source)
            self.compile_target(self.benchmark.target_src, self.benchmark.target_bin)
            print("  Reverted to last verified working source.")

            print("\n" + UI_MAJOR_LINE)
            print("[FINAL REPORT]")
            print(UI_MAJOR_LINE)
            print(f"Baseline execution time        : {baseline_time:.4f}s")
            print(f"Best optimized execution time  : {best_opt_time:.4f}s")
            print(f"Best verified speedup          : {best_speedup_so_far:.4f}x")

        if verify_passed:
            print("Semantic verification           : PASSED")
            print("Status                          : Final optimized code is verified correct.")
        else:
            print("Semantic verification           : FAILED")
            print("Status                          : Reverted to last verified working source.")

        print("\n" + UI_MAJOR_LINE)
        print("[FINAL REPORT]")
        print(UI_MAJOR_LINE)
        print(f"Baseline execution time        : {baseline_time:.4f}s")
        print(f"Best optimized execution time  : {best_opt_time:.4f}s")
        print(f"Best verified speedup          : {best_speedup_so_far:.4f}x")

        append_summary_row({
            "benchmark": self.benchmark.name,
            "model": self.model,
            "baseline_time": f"{baseline_time:.6f}",
            "best_opt_time": f"{best_opt_time:.6f}",
            "speedup": f"{best_speedup_so_far:.6f}",
            "verification_status": (
                "verification_passed"
                if verify_passed
                else "verification_not_passed"
            ),
            "iterations_used": iterations_used,
        })


if __name__ == "__main__":
    args = parse_args()
    benchmark_configs = load_benchmark_configs(args.benchmark)
    for benchmark in benchmark_configs:
        optimizer = Optimizer(
            benchmark=benchmark,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        optimizer.main_loop()
