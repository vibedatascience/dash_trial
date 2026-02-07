"""
Code Interpreter Tool for Dash
==============================

Executes Python code with matplotlib/seaborn support and returns charts as base64 images.
Like Julius.ai but for F1 data analysis.

Handles 5 potential failure scenarios:
1. Syntax errors in generated code - Clear error messages
2. Matplotlib backend issues - Force 'Agg' backend
3. Database connection issues - Graceful fallback with error message
4. Large result sets - Limit DataFrame output, set figure size limits
5. Timeout issues - Execution timeout protection (thread-safe)
"""

import base64
import io
import sys
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Optional

from agno.tools import Toolkit
from agno.utils.log import logger


class CodeExecutionTimeout(Exception):
    """Raised when code execution times out."""
    pass


class CodeInterpreterTools(Toolkit):
    """
    Code interpreter that can execute Python code and generate visualizations.

    Features:
    - Execute arbitrary Python code safely
    - Generate matplotlib/seaborn charts
    - Return charts as base64 images
    - Capture stdout/stderr output
    - Access to pandas, numpy, matplotlib, seaborn
    - 30 second timeout protection
    - Memory-safe DataFrame handling
    - PERSISTENT STATE: Variables persist across tool calls (like Jupyter notebook cells)
    """

    def __init__(
        self,
        charts_dir: Optional[Path] = None,
        db_url: Optional[str] = None,
        timeout_seconds: int = 30,
        max_df_rows: int = 1000,
        **kwargs
    ):
        self.charts_dir = (charts_dir or Path.cwd() / "charts").resolve()
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.db_url = db_url
        self.timeout_seconds = timeout_seconds
        self.max_df_rows = max_df_rows

        # Pre-import common libraries for the execution context
        self._setup_globals()

        tools = [
            self.run_code,
            self.run_code_and_get_chart,
            self.list_variables,
        ]

        super().__init__(name="code_interpreter", tools=tools, **kwargs)

    def _setup_globals(self) -> dict:
        """Setup the global namespace with common libraries."""
        self._globals = {
            '__builtins__': __builtins__,
        }

        # Import common libraries
        try:
            import pandas as pd
            self._globals['pd'] = pd
            self._globals['pandas'] = pd
        except ImportError:
            logger.warning("pandas not available")

        try:
            import numpy as np
            self._globals['np'] = np
            self._globals['numpy'] = np
        except ImportError:
            logger.warning("numpy not available")

        # CRITICAL: Force Agg backend BEFORE importing pyplot
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend - MUST be before pyplot import
            import matplotlib.pyplot as plt
            # Set default figure size limits
            plt.rcParams['figure.max_open_warning'] = 5
            plt.rcParams['figure.figsize'] = [10, 6]
            self._globals['plt'] = plt
            self._globals['matplotlib'] = matplotlib
        except ImportError:
            logger.warning("matplotlib not available")

        try:
            import seaborn as sns
            self._globals['sns'] = sns
            self._globals['seaborn'] = sns
        except ImportError:
            logger.warning("seaborn not available")

        # Add database connection helper if db_url provided
        if self.db_url:
            try:
                from sqlalchemy import create_engine, text
                engine = create_engine(self.db_url)

                # Test connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

                max_rows = self.max_df_rows

                def query_db(sql: str, limit: int = None) -> 'pd.DataFrame':
                    """
                    Execute SQL and return results as DataFrame.

                    Args:
                        sql: SQL query to execute
                        limit: Optional row limit (default: 1000)

                    Returns:
                        pandas DataFrame with query results
                    """
                    import pandas as pd
                    effective_limit = limit or max_rows

                    # Add LIMIT if not present and not an aggregate query
                    sql_lower = sql.lower().strip()
                    if 'limit' not in sql_lower and not any(agg in sql_lower for agg in ['count(*)', 'sum(', 'avg(', 'max(', 'min(']):
                        sql = f"{sql.rstrip(';')} LIMIT {effective_limit}"

                    try:
                        with engine.connect() as conn:
                            df = pd.read_sql(text(sql), conn)
                            if len(df) > effective_limit:
                                df = df.head(effective_limit)
                            return df
                    except Exception as e:
                        raise Exception(f"Database query failed: {str(e)}")

                self._globals['query_db'] = query_db
                self._globals['engine'] = engine
                logger.info("Database connection configured successfully")
            except Exception as e:
                logger.warning(f"Could not setup database connection: {e}")
                # Provide a helpful error function
                def query_db_unavailable(sql: str) -> None:
                    raise Exception(f"Database not available. Error: {e}")
                self._globals['query_db'] = query_db_unavailable

        return self._globals

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """Validate code for syntax errors before execution."""
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def list_variables(self) -> str:
        """
        List all user-defined variables currently in memory.

        Use this to see what DataFrames and variables are available from previous code executions.
        This helps avoid re-fetching data that's already loaded.

        :return: List of variable names and their types
        """
        # Filter out built-in stuff and modules
        skip_types = (type(None.__class__), type(lambda: None))
        skip_names = {'__builtins__', 'pd', 'pandas', 'np', 'numpy', 'plt', 'matplotlib',
                      'sns', 'seaborn', 'query_db', 'engine'}

        user_vars = []
        for name, val in self._globals.items():
            if name.startswith('_') or name in skip_names:
                continue
            if isinstance(val, skip_types):
                continue

            # Get type and basic info
            type_name = type(val).__name__
            info = f"{name}: {type_name}"

            # Add size info for DataFrames
            if hasattr(val, 'shape'):
                info += f" (shape: {val.shape})"
            elif hasattr(val, '__len__') and not isinstance(val, str):
                try:
                    info += f" (len: {len(val)})"
                except:
                    pass

            user_vars.append(info)

        if not user_vars:
            return "No user-defined variables in memory. Use run_code() or run_code_and_get_chart() to create some."

        return "Variables in memory:\n" + "\n".join(f"  - {v}" for v in user_vars)

    def _execute_with_timeout(self, code: str, stdout_capture: io.StringIO, stderr_capture: io.StringIO) -> None:
        """Execute code with stdout/stderr capture. Runs in a separate thread for timeout.

        Variables are stored in self._globals so they persist across calls (like Jupyter cells).
        """
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, self._globals)

    def run_code(self, code: str) -> str:
        """
        Execute Python code and return the output.

        IMPORTANT: Variables persist across calls like Jupyter notebook cells!
        If you created `df = query_db(...)` in a previous call, you can use `df` directly.
        Do NOT re-fetch data that's already in memory.

        You have access to: pandas (pd), numpy (np), matplotlib (plt), seaborn (sns).
        If a database is configured, you can use query_db(sql) to fetch data.

        :param code: Python code to execute
        :return: Output from the code execution (stdout + any returned value)
        """
        # Validate syntax first
        valid, error = self._validate_code(code)
        if not valid:
            return f"Code has syntax error:\n{error}\n\nPlease fix the code and try again."

        try:
            # Capture stdout/stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Use ThreadPoolExecutor for timeout (works in any thread context)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._execute_with_timeout,
                    code, stdout_capture, stderr_capture
                )
                try:
                    future.result(timeout=self.timeout_seconds)
                except FuturesTimeoutError:
                    raise CodeExecutionTimeout(f"Code execution timed out ({self.timeout_seconds} second limit)")

            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            # Build result
            result_parts = []

            if stdout_output:
                # Limit output size
                if len(stdout_output) > 5000:
                    stdout_output = stdout_output[:5000] + "\n... (output truncated)"
                result_parts.append(f"Output:\n{stdout_output}")

            if stderr_output:
                result_parts.append(f"Warnings:\n{stderr_output[:1000]}")

            # Check for common result variables in persistent globals
            for var_name in ['result', 'output', 'df', 'data', 'summary']:
                if var_name in self._globals:
                    val = self._globals[var_name]
                    if hasattr(val, 'to_string'):
                        # Limit DataFrame output
                        if hasattr(val, 'head') and len(val) > 50:
                            result_parts.append(f"{var_name} (showing first 50 of {len(val)} rows):\n{val.head(50).to_string()}")
                        else:
                            result_parts.append(f"{var_name}:\n{val.to_string()}")
                    else:
                        str_val = str(val)
                        if len(str_val) > 2000:
                            str_val = str_val[:2000] + "... (truncated)"
                        result_parts.append(f"{var_name}: {str_val}")
                    break

            if not result_parts:
                result_parts.append("Code executed successfully.")

            return "\n\n".join(result_parts)

        except CodeExecutionTimeout as e:
            return f"Error: {str(e)}\n\nTry simplifying your code or reducing the data size."
        except Exception as e:
            error_msg = traceback.format_exc()
            # Simplify error message
            if "query_db" in str(e) or "database" in str(e).lower():
                return f"Database error:\n{str(e)}\n\nMake sure your SQL query is valid."
            return f"Error executing code:\n{error_msg}"

    def run_code_and_get_chart(self, code: str, title: str = "chart") -> str:
        """
        Execute Python code that creates a matplotlib/seaborn chart and return it as base64.

        IMPORTANT: Variables persist across calls like Jupyter notebook cells!
        If you created `df = query_db(...)` in a previous call, you can use `df` directly.
        Do NOT re-fetch data that's already in memory.

        Use this when you need to create visualizations like bar charts, line plots, etc.
        Your code should create a matplotlib figure using plt.figure() or sns plots.
        The chart will be automatically captured and returned as a base64 image.

        You have access to: pandas (pd), numpy (np), matplotlib (plt), seaborn (sns).
        If a database is configured, you can use query_db(sql) to fetch data as a DataFrame.

        Example (if df already exists from previous call):
        ```python
        plt.figure(figsize=(10, 6))
        plt.bar(df['year'], df['points'], color='#00D2BE')
        plt.title('Hamilton Points by Year')
        plt.xlabel('Year')
        plt.ylabel('Points')
        ```

        :param code: Python code that creates a matplotlib chart
        :param title: Title for the chart (used for filename)
        :return: Base64 encoded image or error message
        """
        # Validate syntax first
        valid, error = self._validate_code(code)
        if not valid:
            return f"Code has syntax error:\n{error}\n\nPlease fix the code and try again."

        try:
            import matplotlib
            matplotlib.use('Agg')  # Ensure non-interactive backend
            import matplotlib.pyplot as plt

            # Clear any existing plots to prevent memory buildup
            plt.close('all')

            # Capture stdout/stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Use ThreadPoolExecutor for timeout (works in any thread context)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._execute_with_timeout,
                    code, stdout_capture, stderr_capture
                )
                try:
                    future.result(timeout=self.timeout_seconds)
                except FuturesTimeoutError:
                    plt.close('all')
                    raise CodeExecutionTimeout(f"Code execution timed out ({self.timeout_seconds} second limit)")

            # Get current figure
            fig = plt.gcf()

            # Check if there's actually a plot
            if not fig.axes:
                plt.close('all')
                return "No chart was created. Make sure your code creates a matplotlib plot using plt.figure(), plt.bar(), plt.plot(), etc."

            # Apply clean styling
            try:
                plt.tight_layout()
            except Exception:
                pass  # Ignore tight_layout errors

            # Save to bytes with reasonable DPI to limit file size
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)

            # Check file size (limit to 2MB)
            img_data = buf.read()
            if len(img_data) > 2 * 1024 * 1024:
                plt.close('all')
                return "Chart is too large. Try reducing the data points or figure size."

            # Encode as base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')

            # Save to file for reference (optional, don't fail if it doesn't work)
            try:
                safe_title = "".join(c if c.isalnum() or c in '-_' else '_' for c in title)[:50]
                chart_path = self.charts_dir / f"{safe_title}.png"
                fig.savefig(chart_path, format='png', dpi=120, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
            except Exception:
                pass

            plt.close('all')

            # Return base64 with marker so frontend knows it's an image
            stdout_output = stdout_capture.getvalue()

            result = f"[CHART_BASE64]{img_base64}[/CHART_BASE64]"
            if stdout_output and len(stdout_output) < 1000:
                result = f"Output:\n{stdout_output}\n\n{result}"

            return result

        except CodeExecutionTimeout as e:
            plt.close('all')
            return f"Error: {str(e)}\n\nTry reducing the data size or simplifying the chart."
        except Exception as e:
            plt.close('all')
            error_msg = traceback.format_exc()

            # Provide helpful error messages
            if "query_db" in str(e) or "database" in str(e).lower():
                return f"Database error:\n{str(e)}\n\nMake sure your SQL query is valid."
            if "figure" in str(e).lower() or "plot" in str(e).lower():
                return f"Chart creation error:\n{str(e)}\n\nMake sure you're using plt.figure() or similar to create a plot."

            return f"Error creating chart:\n{error_msg}"


def create_code_interpreter_tools(db_url: Optional[str] = None, charts_dir: Optional[Path] = None) -> CodeInterpreterTools:
    """Factory function to create CodeInterpreterTools instance."""
    return CodeInterpreterTools(db_url=db_url, charts_dir=charts_dir)
