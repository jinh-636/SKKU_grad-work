import requests
from rich.text import Text

from cerebrum.config.config_manager import config


class FileOperationConfirmationHandler:
    """Handle kernel confirmation responses in either AIOS terminal entry point."""
    def _submit_operation_decision(self, response, decision):
        result = requests.post(
            f"{config.get('kernel', 'base_url').rstrip('/')}/operations/{response['request_id']}/decision",
            json={"confirmation_id": response["confirmation_id"], "decision": decision},
            timeout=(10, float(config.get("kernel", "timeout") or 30)),
        )
        if not result.ok:
            try:
                detail = result.json().get("detail", result.reason)
            except ValueError:
                detail = result.reason
            raise RuntimeError(f"Kernel HTTP {result.status_code}: {detail}")
        return result.json()

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
        self.console.print(Text(f"Request: {response['request_id']}"))
        self.console.print(Text(f"Confirmation: {response['confirmation_id']}"))
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
