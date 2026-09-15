import requests
from rich.text import Text

from cerebrum.config.config_manager import config


class FileOperationConfirmationHandler:
    """Handle kernel confirmation responses in either AIOS terminal entry point."""
    def _submit_operation_decision(self, response, decision):
        payload = decision if isinstance(decision, dict) else {"decision": decision}
        result = requests.post(
            f"{config.get('kernel', 'base_url').rstrip('/')}/operations/{response['request_id']}/decision",
            json={"confirmation_id": response["confirmation_id"], **payload},
            timeout=(10, float(config.get("kernel", "timeout") or 30)),
        )
        if not result.ok:
            try:
                detail = result.json().get("detail", result.reason)
            except ValueError:
                detail = result.reason
            raise RuntimeError(f"Kernel HTTP {result.status_code}: {detail}")
        return result.json()

    def _prompt_duplicate_selection(self, details):
        delete_ids = {}
        self.console.print(Text("Select file numbers to delete. Use commas for multiple files, "
                                "all to delete every file in this group, or no to cancel."))
        for group in details["groups"]:
            self.console.print(Text(f"Duplicate group {group['id']}", style="bold yellow"))
            numbers = {item["id"]: index for index, item in enumerate(group["files"], 1)}
            for item in group["files"]:
                self.console.print(Text(f"  {numbers[item['id']]}. {item['file_path']} "
                                        f"({item['size_bytes']} bytes; modified {item['modified_at']})"))
            for pair in group["pairs"]:
                self.console.print(Text(f"  Similarity {numbers[pair['left']]} <-> "
                                        f"{numbers[pair['right']]}: {pair['similarity']:.6f}"))
            while True:
                choice = self.session.prompt(f"Delete in {group['id']} [numbers/all/no]: ").strip().lower()
                if choice in ("n", "no", "cancel"):
                    return "cancel"
                if choice == "all":
                    selected = list(range(1, len(group["files"]) + 1))
                else:
                    try:
                        selected = [int(value.strip()) for value in choice.split(",")]
                    except ValueError:
                        selected = []
                if (selected and len(set(selected)) == len(selected)
                        and all(1 <= index <= len(group["files"]) for index in selected)):
                    delete_ids[group["id"]] = [group["files"][index - 1]["id"] for index in selected]
                    break
                self.console.print(Text("Enter valid file numbers, all, or no."))
        return {"decision": "select", "delete_ids": delete_ids}

    def _prompt_operation_decision(self, response):
        operation = response["operation"]
        self.console.print(Text("[Confirmation required]", style="bold yellow"))
        self.console.print(Text(f"Operation: {operation['name']}"))
        self.console.print(Text(f"File: {operation.get('file_path', '')}"))
        self.console.print(Text(
            f"Completed: {response['completed_count']}/{response['total_count']}"
        ))
        if "content_preview" in operation:
            self.console.print(Text(f"Content ({operation['size_bytes']} bytes):"))
            self.console.print(Text(operation["content_preview"]))
            if operation.get("content_truncated"):
                self.console.print(Text("[Preview limited to 1000 characters]"))
        self.console.print(Text(response["message"]))
        details = response.get("deduplication")
        if details is not None:
            self.console.print(Text(f"Root: {details['root_dir']} | Cosine threshold: {details['threshold']} "
                                    f"| Compared: {details['scanned_count']} files"))
            for skipped in details["skipped"]:
                self.console.print(Text(f"Skipped: {skipped['file_path']} ({skipped['reason']})"))
            if details["stage"] == "select":
                return self._prompt_duplicate_selection(details)
            self.console.print(Text("Retain:", style="bold green"))
            if not details["retained"]:
                self.console.print(Text("  (none)"))
            for item in details["retained"]:
                self.console.print(Text(f"  {item['file_path']}"))
            self.console.print(Text("Delete after confirmation:", style="bold red"))
            for item in details["deletions"]:
                self.console.print(Text(f"  {item['file']['file_path']} ({item['file']['size_bytes']} bytes)"))
            self.console.print(Text(f"Selected: {len(details['deletions'])} files, "
                                    f"{sum(item['file']['size_bytes'] for item in details['deletions'])} bytes"))
        while True:
            choice = self.session.prompt("Proceed? [y/n]: ").strip().lower()
            if choice in ("y", "yes"):
                return "approve"
            if choice in ("n", "no"):
                return "cancel"
            self.console.print("Please enter y or n.")

    def _handle_operation_response(self, response):
        while isinstance(response, dict) and response.get("status") == "needs_confirmation":
            interrupted = None
            try:
                decision = self._prompt_operation_decision(response)
            except (KeyboardInterrupt, EOFError) as error:
                interrupted = error
                decision = "cancel"
            response = self._submit_operation_decision(response, decision)
            if isinstance(interrupted, EOFError):
                self.console.print(Text(response.get("message", "Cancelled."), style="yellow"))
                raise interrupted

        if isinstance(response, str):
            self.console.print(Text(response, style="bold green"))
        elif isinstance(response, dict):
            style = "red" if response.get("status") == "failed" else "yellow"
            self.console.print(Text(str(response.get("message", response)), style=style))
            for result in response.get("results", []):
                self.console.print(Text(str(result["result"])))
        else:
            self.console.print(Text("The kernel returned an invalid file operation response.", style="red"))
