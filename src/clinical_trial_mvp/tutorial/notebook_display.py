"""JSON views with readable fallbacks for GitHub and other static notebook viewers."""
import json
import re

from IPython.core.display import JSON as IPythonJSON


def json_text(data):
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def json_markdown(data):
    text = json_text(data)
    # A prompt can itself contain code fences. Keep those inside the JSON block.
    longest = max((len(run) for run in re.findall(r'`+', text)), default=0)
    fence = '`' * max(3, longest + 1)
    return fence + 'json\n' + text + '\n' + fence


class JSON(IPythonJSON):
    """Retain Jupyter's JSON tree and add static Markdown/plain-text renderings."""

    def _repr_pretty_(self, printer, cycle):
        printer.text(json_text(self.data))

    def _repr_markdown_(self):
        return json_markdown(self.data)
