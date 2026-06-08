from __future__ import annotations

from pathlib import Path

from ..lang.ast import Program
from ..lang.parser import parse_program
from ..reporter import CliReporter, Reporter
from ..source_loading import merge_source_documents
from ..tests_runner import run_tests
from .interfaces import CodegenAgent, CodegenConfig, ProgramContext
from .translator import (
    kedi_procedure_to_python_signature,
    kedi_signature_to_python_signature,
    kedi_typedef_to_python_class,
    python_impl_to_kedi_impl,
    python_tests_to_kedi_tests,
)


class CodegenManager:
    def __init__(
        self,
        agent: CodegenAgent,
        config: CodegenConfig | None = None,
        use_cache: bool = True,
        reporter: Reporter | None = None,
    ) -> None:
        self.agent = agent
        self.config = config or CodegenConfig()
        self.use_cache = use_cache
        self._reporter = reporter or CliReporter()

    def cache_path_for(self, source_path: Path) -> Path:
        return source_path.with_suffix(".cache.kedi")

    def is_cached(self, source_path: Path, proc_name: str) -> bool:
        if not self.use_cache:
            return False
        cache = self.cache_path_for(source_path)
        if not cache.exists():
            return False
        text = cache.read_text(encoding="utf-8")
        return f"@{proc_name}(" in text

    def write_cache(self, source_path: Path, tests: str, impl: str) -> None:
        # Always write cache for execution (needed for tests and program run)
        # The caller will clean it up if use_cache=False
        cache = self.cache_path_for(source_path)
        content = tests.strip() + "\n\n" + impl.strip() + "\n"
        cache.write_text(content, encoding="utf-8")

    def extract_context_from_ast(self, program: Program) -> ProgramContext:
        """Extract structured context from parsed AST as Python code."""
        context = ProgramContext()

        # Prelude (Python code at the top) - keep as-is
        if program.prelude:
            context.preamble = program.prelude

        # Type definitions - convert to Python BaseModel classes
        context.type_definitions = [kedi_typedef_to_python_class(td) for td in program.type_defs]

        # Procedure definitions (exclude AI-generated ones) - convert to Python function signatures
        context.procedure_definitions = [
            kedi_procedure_to_python_signature(pd)
            for pd in program.procedures
            if not pd.auto_spec and not getattr(pd, "ai_prompt", None)
        ]

        return context

    def parse_with_cache(self, source_path: Path) -> Program:
        """Parse source file, merging with cache if it exists."""
        documents = [(source_path.resolve(), source_path.read_text(encoding="utf-8"))]
        cache = self.cache_path_for(source_path)
        if cache.exists():
            documents.append((cache.resolve(), cache.read_text(encoding="utf-8")))
        loaded = merge_source_documents(documents)
        return parse_program(
            loaded.text,
            source_path=str(source_path.resolve()),
            source_map=loaded.source_map,
        )

    def verify_compilation(self, source_path: Path) -> str | None:
        """Verify the code compiles. Returns error message if it doesn't."""
        try:
            self.parse_with_cache(source_path)
            return None
        except Exception as e:
            return str(e)

    def run_tests_and_get_errors(self, source_path: Path) -> str | None:
        """Run tests. Returns error message if they fail."""
        try:
            results = run_tests(str(source_path))
            failed = [c for s in results for c in s.cases if not c.success]
            if failed:
                return "\n".join(c.error or f"Test {c.name} failed" for c in failed)
            return None
        except Exception as e:
            return str(e)

    def context_for_generation(self, source_path: Path) -> ProgramContext:
        """Extract source context, falling back to an empty context on parse errors."""
        try:
            program = parse_program(source_path.read_text(encoding="utf-8"))
            return self.extract_context_from_ast(program)
        except Exception:
            return ProgramContext()

    def cleanup_failed_cache(self, source_path: Path) -> None:
        cache = self.cache_path_for(source_path)
        if cache.exists():
            cache.unlink()

    def orchestrate_generation(
        self, source_path: Path, proc_header: str, prompt: str, spec: str
    ) -> bool:
        """Orchestrate Python code generation and translation to Kedi DSL.

        Iterative Flow (N stages, where N = max_retries):
        Stage 1: Try tests 1 time → Try impl 1 time
        Stage 2: Try tests 2 times → Try impl 2 times
        Stage 3: Try tests 3 times → Try impl 3 times
        ...
        Stage N: Try tests N times → Try impl N times

        At each stage:
        1. Try generating tests up to stage_number times (with error feedback)
        2. If all test attempts fail, move to next stage
        3. Try generating implementation up to stage_number times (with error feedback)
        4. If all impl attempts fail, move to next stage
        5. If any impl succeeds and passes tests, return True
        """
        name = proc_header.split("(")[0].strip().lstrip("@")
        kedi_tests = ""
        python_tests = ""

        if self.is_cached(source_path, name):
            return True

        context = self.context_for_generation(source_path)

        # Convert Kedi signature to Python signature
        python_sig = kedi_signature_to_python_signature(proc_header)

        self._reporter.emit(f"Code generation in progress for {name}", end="", flush=True)

        # Iterative generation: each stage gets more attempts
        test_error_feedback = None
        impl_error_feedback = None

        for stage in range(self.config.max_retries):
            num_attempts = stage + 1  # Stage 1 → 1 attempt, Stage 2 → 2 attempts, etc.

            # Try generating tests (num_attempts times)
            test_success = False
            for test_attempt in range(num_attempts):
                self._reporter.progress_char()
                python_tests = self.agent.generate_tests(
                    python_sig, prompt, context, test_error_feedback
                )

                # Translate tests to Kedi
                kedi_tests = python_tests_to_kedi_tests(python_tests, name)

                # Verify tests compile
                self.write_cache(source_path, kedi_tests, "")
                test_compile_error = self.verify_compilation(source_path)

                if not test_compile_error:
                    test_success = True
                    break

                test_error_feedback = f"Previous Python tests:\n{python_tests}\n\nKedi translation:\n{kedi_tests}\n\nError: {test_compile_error}"

            if not test_success:
                continue

            # Tests compiled, now try generating implementation (num_attempts times)
            impl_success = False
            for impl_attempt in range(num_attempts):
                self._reporter.progress_char()
                python_impl = self.agent.generate_impl(
                    python_sig, prompt, context, python_tests, impl_error_feedback
                )

                # Translate implementation to Kedi
                kedi_impl = python_impl_to_kedi_impl(python_impl, proc_header)

                self.write_cache(source_path, kedi_tests, kedi_impl)

                # Verify implementation compiles
                impl_compile_error = self.verify_compilation(source_path)
                if impl_compile_error:
                    impl_error_feedback = f"Previous Python implementation:\n{python_impl}\n\nKedi translation:\n{kedi_impl}\n\nCompilation error: {impl_compile_error}"
                    continue

                # Implementation compiled, run tests
                test_run_error = self.run_tests_and_get_errors(source_path)
                if not test_run_error:
                    self._reporter.emit(" ✓", flush=True)
                    impl_success = True
                    break

                impl_error_feedback = f"Previous Python implementation:\n{python_impl}\n\nKedi translation:\n{kedi_impl}\n\nTest errors: {test_run_error}"

            if impl_success:
                return True

            # Implementation failed all attempts, prepare feedback for next stage
            test_error_feedback = f"Previous tests may have been too strict or incorrect.\nLast implementation error: {impl_error_feedback}"

        self._reporter.emit(" ✗", flush=True)
        # Clean up failed cache - don't persist code that doesn't pass tests
        self.cleanup_failed_cache(source_path)
        return False
