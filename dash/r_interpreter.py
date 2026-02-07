"""
R Code Interpreter
==================

Execute R code with persistent state and chart generation.
Separate from Python interpreter to keep concerns isolated.
"""

import subprocess
import tempfile
import os
import base64
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RInterpreter:
    """Execute R code with persistent state across calls."""

    def __init__(self, db_url: str | None = None):
        self._db_url = db_url
        self._temp_dir = tempfile.mkdtemp(prefix="dash_r_")
        self._state_file = os.path.join(self._temp_dir, "state.RData")
        self._initialized = False

        # Initialize R environment with DB connection and common packages
        self._init_r_env()

    def _init_r_env(self):
        """Initialize R environment — write preamble script and create initial state."""
        # Build preamble that gets sourced at the start of EVERY script.
        # This is necessary because each Rscript subprocess is a fresh process,
        # and save.image()/load() only persist objects, NOT loaded packages.
        preamble = """# --- Dash R preamble (sourced every call) ---
# Reload packages (these don't survive save.image/load)
suppressPackageStartupMessages({
    library(dplyr)
    library(tidyr)
    library(readr)
    library(ggplot2)
    library(stringr)
    library(purrr)
})

# Use minimal theme as default — agent controls all styling via prompt
theme_set(theme_minimal())
"""

        # Add DB connection to preamble
        if self._db_url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self._db_url)
                preamble += f"""
# Database connection (re-established each call since handles don't serialize)
tryCatch({{
    suppressPackageStartupMessages({{
        library(DBI)
        library(RPostgres)
    }})
    db_con <<- dbConnect(
        RPostgres::Postgres(),
        host = "{parsed.hostname or 'localhost'}",
        port = {parsed.port or 5432},
        dbname = "{parsed.path.lstrip('/') if parsed.path else 'ai'}",
        user = "{parsed.username or 'ai'}",
        password = "{parsed.password or 'ai'}"
    )
    query_db <<- function(sql) dbGetQuery(db_con, sql)
    run_sql <<- query_db
}}, error = function(e) {{
    query_db <<- function(sql) stop("Database not available. Use Python's run_sql tool instead.")
    run_sql <<- query_db
}})
"""
            except Exception as e:
                logger.warning(f"Could not set up R DB connection: {e}")

        # Write preamble to a file that gets sourced each call
        self._preamble_file = os.path.join(self._temp_dir, "preamble.R")
        with open(self._preamble_file, 'w') as f:
            f.write(preamble)

        # Run preamble once to create initial state.RData
        result = self._run_r_code("# Initial setup", save_state=True)
        if "Error" not in result:
            self._initialized = True
        else:
            logger.warning(f"R initialization warning: {result}")

    def _run_r_code(self, code: str, save_state: bool = True, load_state: bool = True) -> str:
        """Execute R code via Rscript with persistent state.

        Each call spawns a fresh Rscript process. Persistence works via:
        1. Preamble file sourced every call (reloads packages, theme, DB connection)
        2. state.RData loaded to restore user variables from previous calls
        3. User code wrapped in tryCatch so save.image() runs even on error
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False, dir=self._temp_dir) as f:
            script = ""

            if load_state and hasattr(self, '_preamble_file'):
                # Source preamble (reloads packages, theme, DB — these don't persist via .RData)
                script += f'source("{self._preamble_file}", local = FALSE)\n'
                # Restore user variables from previous calls
                if os.path.exists(self._state_file):
                    script += f'load("{self._state_file}")\n'

            if save_state:
                # Wrap user code in tryCatch so save.image() ALWAYS runs,
                # even if the user code errors out. Without this, an error
                # in line N causes R to exit before save.image(), losing all
                # variables created in lines 1..N-1.
                script += f'.dash_err <- tryCatch({{\n'
                script += code + "\n"
                script += '  NULL\n'
                script += '}, error = function(e) e)\n'
                # Clean up internal var, save state, then re-signal error
                script += '.dash_err_msg <- if (inherits(.dash_err, "error")) .dash_err$message else NULL\n'
                script += 'rm(.dash_err)\n'
                script += f'save.image(file = "{self._state_file}")\n'
                script += 'if (!is.null(.dash_err_msg)) stop(.dash_err_msg, call. = FALSE)\n'
            else:
                script += code + "\n"

            f.write(script)
            f.flush()

            try:
                result = subprocess.run(
                    ['Rscript', '--vanilla', f.name],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self._temp_dir
                )

                output = result.stdout
                errors = result.stderr

                # Filter out common R messages that aren't real errors
                error_lines = []
                for line in errors.split('\n'):
                    if line.strip() and not any(skip in line for skip in [
                        'Attaching package',
                        'The following objects are masked',
                        'Loading required package',
                        'Registered S3 method',
                        'was built under R version',
                        'Use `spec()` to retrieve',
                        'Specify the column types',
                        'show_col_types = FALSE',
                        'Warning message:',
                        'In addition:',
                    ]):
                        error_lines.append(line)

                filtered_errors = '\n'.join(error_lines).strip()

                if result.returncode != 0:
                    return f"R Error:\n{filtered_errors or output}"

                if filtered_errors:
                    return f"Output:\n{output}\n\nWarnings:\n{filtered_errors}"

                return output.strip() if output.strip() else "Code executed successfully (no output)"

            except subprocess.TimeoutExpired:
                return "Error: R code execution timed out (120s limit)"
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}"
            finally:
                os.unlink(f.name)

    def run_code(self, code: str) -> str:
        """Execute R code and return output."""
        return self._run_r_code(code)

    def run_code_and_get_chart(self, code: str) -> str:
        """Execute R code that creates a ggplot2/base R chart, return base64 image."""
        chart_file = os.path.join(self._temp_dir, f"chart_{os.getpid()}.png")

        # Wrap code to save plot
        wrapped_code = f"""
# User code
{code}

# Save the last plot
ggsave("{chart_file}", width = 10, height = 6, dpi = 150)
"""

        result = self._run_r_code(wrapped_code)

        # Check if chart was created
        if os.path.exists(chart_file):
            try:
                with open(chart_file, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                os.unlink(chart_file)
                return f"[CHART_BASE64]{img_base64}[/CHART_BASE64]"
            except Exception as e:
                return f"Chart Error: Could not read chart file: {e}"
        else:
            # Maybe it's a base R plot, try different approach
            base_r_code = f"""
png("{chart_file}", width = 1000, height = 600, res = 150)
{code}
dev.off()
"""
            result = self._run_r_code(base_r_code, save_state=False)

            if os.path.exists(chart_file):
                try:
                    with open(chart_file, 'rb') as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                    os.unlink(chart_file)
                    return f"[CHART_BASE64]{img_base64}[/CHART_BASE64]"
                except Exception as e:
                    return f"Chart Error: Could not read chart file: {e}"

            return f"Chart Error: No chart was generated. R output:\n{result}"

    def list_variables(self) -> str:
        """List all user-defined variables in the R session."""
        code = """
vars <- ls()
# Filter out functions and internal objects
user_vars <- vars[!sapply(vars, function(x) is.function(get(x)) || startsWith(x, "."))]
if (length(user_vars) == 0) {
    cat("No user-defined variables in session.")
} else {
    for (v in user_vars) {
        obj <- get(v)
        type <- class(obj)[1]
        if (is.data.frame(obj)) {
            cat(sprintf("%s: data.frame (%d x %d)\\n", v, nrow(obj), ncol(obj)))
        } else if (is.vector(obj)) {
            cat(sprintf("%s: %s (length %d)\\n", v, type, length(obj)))
        } else {
            cat(sprintf("%s: %s\\n", v, type))
        }
    }
}
"""
        return self._run_r_code(code, save_state=False)

    def get_variable_as_json(self, var_name: str) -> dict | list | None:
        """Get an R variable as JSON (for D3 charts)."""
        code = f"""
if (!exists("{var_name}")) {{
    cat("__ERROR__: Variable not found")
}} else {{
    library(jsonlite)
    cat(toJSON({var_name}, dataframe = "rows"))
}}
"""
        result = self._run_r_code(code, save_state=False)

        if "__ERROR__" in result:
            return None

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return None

    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self._temp_dir)
        except Exception as e:
            logger.warning(f"Could not clean up R temp dir: {e}")


# Tool definitions for R
R_TOOLS = [
    {
        "name": "run_r_code",
        "description": """Execute R code for data analysis and manipulation.

Use this tool when the user prefers R or asks for R-specific analysis.

The execution environment persists between calls (variables are saved).
Pre-loaded packages: dplyr, ggplot2, tidyr, DBI, RPostgres
Pre-loaded functions: query_db(sql), run_sql(sql) - query the database

Example:
```r
df <- query_db("SELECT * FROM drivers_championship WHERE year = 2020")
df %>%
    group_by(team) %>%
    summarize(total_points = sum(points)) %>%
    arrange(desc(total_points))
```

IMPORTANT: Store results in variables to use in later code cells.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "R code to execute."
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "run_r_chart",
        "description": """Execute R code that creates a ggplot2 visualization.

Use this tool when user wants R/ggplot2 charts.
A dark theme is applied automatically.

Pre-loaded: ggplot2, dplyr, tidyr, query_db(sql)

Example:
```r
df <- query_db("SELECT year, points FROM drivers_championship WHERE name = 'Lewis Hamilton'")
ggplot(df, aes(x = year, y = points)) +
    geom_line(color = "#f97316", size = 1.2) +
    geom_point(color = "#f97316", size = 3) +
    labs(title = "Hamilton's Championship Points", x = "Year", y = "Points")
```

The chart is returned as a base64-encoded image.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "R code that creates a ggplot2 or base R chart."
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "list_r_variables",
        "description": "List all variables currently defined in the R session. Use this to see what data you have already loaded in R.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]
