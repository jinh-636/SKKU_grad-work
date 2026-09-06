"""Local terminal commands that do not need language-model inference."""

import requests
from rich.table import Table
from rich.text import Text


def show_quota(console, base_url: str, agent_name: str = "terminal") -> None:
    """Display the running kernel's quota snapshot using its read-only API."""
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/storage/quota",
            params={"agent_name": agent_name},
            timeout=10,
        )
        if response.status_code == 404:
            console.print(Text("Quota endpoint not found. Restart the kernel with the updated code.", style="red"))
            return
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        console.print(Text(f"Unable to query quota: {error}", style="red"))
        return
    except ValueError:
        console.print(Text("The kernel returned an invalid quota response.", style="red"))
        return

    try:
        console.print(Text(f"Storage root: {data['root_dir']}"))
        if not data["enabled"]:
            console.print(Text("Semantic quota is disabled.", style="yellow"))
            return
        table = Table(title=Text(f"Semantic Quota - {data['agent_name']}"))
        table.add_column("Category", style="cyan")
        table.add_column("Used (B)", justify="right")
        table.add_column("Limit (B)", justify="right")
        table.add_column("Remaining (B)", justify="right")
        table.add_column("Status")
        for row in data["categories"]:
            limit = row["limit_bytes"]
            remaining = row["remaining_bytes"]
            style = "red" if row["status"] == "exceeded" else "yellow" if row["status"] == "full" else None
            table.add_row(
                Text(row["category"]),
                f"{row['used_bytes']:,}",
                "Unlimited" if limit is None else f"{limit:,}",
                "-" if remaining is None else f"{remaining:,}",
                Text(row["status"].upper(), style=style),
            )
        console.print(table)
        console.print(Text(f"Total used: {data['total_used_bytes']:,} B"))
    except (KeyError, TypeError, ValueError):
        console.print(Text("The kernel returned an invalid quota response.", style="red"))
