import time
import json
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4
from typing import Dict, List, Any, Optional

# Update import to use the new location
from aios.memory.note import MemoryNote
from aios.syscall import Syscall
from aios.syscall.llm import LLMSyscall
from aios.syscall.storage import StorageSyscall, storage_syscalls
from aios.syscall.tool import ToolSyscall
from aios.syscall.memory import MemorySyscall
from aios.hooks.stores._global import (
    global_llm_req_queue_add_message,
    global_memory_req_queue_add_message,
    global_storage_req_queue_add_message,
    global_tool_req_queue_add_message,
    # global_llm_req_queue,
    # global_memory_req_queue,
    # global_storage_req_queue,
    # global_tool_req_queue,
)

import threading

from aios.hooks.types.llm import LLMRequestQueue
from aios.hooks.types.memory import MemoryRequestQueue
from aios.hooks.types.storage import StorageRequestQueue
from aios.hooks.types.tool import ToolRequestQueue

from cerebrum.llm.apis import LLMQuery, LLMResponse
from cerebrum.memory.apis import MemoryQuery, MemoryResponse
from cerebrum.storage.apis import StorageQuery, StorageResponse
from aios.storage.deduplication import ApplyDeduplicationQuery, build_deletion_plan
from cerebrum.tool.apis import ToolQuery, ToolResponse

@dataclass
class FileOperationState:
    """An in-memory workflow checkpoint."""

    request_id: str
    agent_name: str
    operations: List[Dict[str, Any]]
    next_index: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    confirmation_id: Optional[str] = None
    deduplication_scan: Optional[dict] = None
    deletion_plan: Optional[dict] = None


class OperationDecisionError(ValueError):
    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


class SyscallExecutor:
    """
    A class that handles system call execution for different types of operations.
    
    This class provides methods to execute system calls for storage, memory,
    tools, and LLM operations, managing their lifecycle and responses.
    
    Example:
        ```python
        executor = SyscallExecutor()
        response = executor.execute_request("agent_1", LLMQuery(...))
        ```
    """
    
    def __init__(self):
        """Initialize the SyscallExecutor."""
        self.id = 0
        self.id_lock = threading.Lock()
        self.pending_operations: Dict[str, FileOperationState] = {}
        self.pending_operations_lock = threading.Lock()
    
    def create_syscall(self, agent_name: str, query) -> Dict[str, Any]:
        """
        Create a syscall object based on the query type.
        """
        if isinstance(query, LLMQuery):
            return LLMSyscall(agent_name, query)
        elif isinstance(query, StorageQuery):
            return StorageSyscall(agent_name, query)
        elif isinstance(query, MemoryQuery):
            return MemorySyscall(agent_name, query)
        elif isinstance(query, ToolQuery):
            return ToolSyscall(agent_name, query)

    def _execute_syscall(self, agent_name: str, query) -> Dict[str, Any]:
        """
        Execute a system call and collect timing metrics.
        
        Args:
            syscall: The system call object to execute
            
        Returns:
            Dict containing response and timing metrics
            
        Example:
            ```python
            syscall = StorageSyscall("agent_1", query)
            result = executor._execute_syscall(syscall)
            # Returns:
            {
                "response": "Operation completed",
                "start_times": [1234567890.123],
                "end_times": [1234567890.234],
                "waiting_times": [0.111],
                "turnaround_times": [0.222]
            }
            ```
        """
        completed_response = ""
        start_times, end_times = [], []
        waiting_times, turnaround_times = [], []

        with self.id_lock:
            self.id += 1
            syscall_id = self.id
        
        while True:
            # syscall = copy.deepcopy(syscall)
            syscall = self.create_syscall(agent_name, query)
            syscall.set_status("active")
            
            current_time = time.time()
            syscall.set_created_time(current_time)
            syscall.set_response(None)

            if not syscall.get_source():
                syscall.set_source(syscall.agent_name)
            
            if not syscall.get_pid():
                syscall.set_pid(syscall_id)
            
            if isinstance(syscall, LLMSyscall):
                global_llm_req_queue_add_message(syscall)
            elif isinstance(syscall, StorageSyscall):
                global_storage_req_queue_add_message(syscall)
            elif isinstance(syscall, MemorySyscall):
                global_memory_req_queue_add_message(syscall)
            elif isinstance(syscall, ToolSyscall):
                global_tool_req_queue_add_message(syscall)
            
            syscall.start()
            syscall.join()
            
            # breakpoint()

            completed_response = syscall.get_response()
            
            if syscall.get_status() == "done":
                break
            
            # breakpoint()
            
            # Calculate timing metrics
            start_time = syscall.get_start_time()
            end_time = syscall.get_end_time()
            waiting_time = start_time - syscall.get_created_time()
            turnaround_time = end_time - syscall.get_created_time()

            start_times.append(start_time)
            end_times.append(end_time)
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)

        return {
            "response": completed_response,
            "start_times": start_times,
            "end_times": end_times,
            "waiting_times": waiting_times,
            "turnaround_times": turnaround_times,
        }

    def execute_storage_syscall(self, agent_name: str, query: StorageQuery) -> Dict[str, Any]:
        """
        Execute a storage system call.
        
        Args:
            agent_name: Name of the agent making the request
            query: Storage query to execute
            
        Returns:
            Dict containing response and timing metrics
            
        Example:
            ```python
            query = StorageQuery(operation_type="read", params={"path": "/tmp/file.txt"})
            result = executor.execute_storage_syscall("agent_1", query)
            ```
        """
        # syscall = StorageSyscall(agent_name, query)
        # syscall.set_target("storage")
        # global_storage_req_queue_add_message(syscall)
        return self._execute_syscall(agent_name, query)

    def execute_memory_syscall(self, agent_name: str, query: MemoryQuery) -> Dict[str, Any]:
        """
        Execute a memory system call.
        
        Args:
            agent_name: Name of the agent making the request
            query: Memory query to execute
            
        Returns:
            Dict containing response and timing metrics
            
        Example:
            ```python
            query = MemoryQuery(operation="store", data={"key": "value"})
            result = executor.execute_memory_syscall("agent_1", query)
            ```
        """
        # syscall = Syscall(agent_name, query)
        # syscall.set_target("memory")
        # global_memory_req_queue_add_message(syscall)
        return self._execute_syscall(agent_name, query)

    def execute_tool_syscall(self, agent_name: str, query: ToolQuery) -> Dict[str, Any]:
        """
        Execute a tool system call.
        
        Args:
            agent_name: Name of the agent making the request
            tool_calls: List of tool calls to execute
            
        Returns:
            Dict containing response and timing metrics
            
        Example:
            ```python
            tool_calls = [{"name": "calculator", "arguments": {"operation": "add", "numbers": [1, 2]}}]
            result = executor.execute_tool_syscall("agent_1", tool_calls)
            ```
        """
        # syscall = ToolSyscall(agent_name, tool_calls)
        # syscall.set_target("tool")
        # global_tool_req_queue_add_message(syscall)
        return self._execute_syscall(agent_name, query)

    def execute_llm_syscall(self, agent_name: str, query: LLMQuery) -> Dict[str, Any]:
        """
        Execute an LLM system call.
        
        Args:
            agent_name: Name of the agent making the request
            query: LLM query to execute
            
        Returns:
            Dict containing response and timing metrics
            
        Example:
            ```python
            query = LLMQuery(messages=[{"role": "user", "content": "Hello"}], action_type="chat")
            result = executor.execute_llm_syscall("agent_1", query)
            ```
        """
        # syscall = LLMSyscall(agent_name=agent_name, query=query)
        # syscall.set_target("llm")
        # global_llm_req_queue_add_message(syscall)
        return self._execute_syscall(agent_name, query)

    def execute_file_operation(
        self, agent_name: str, query: LLMQuery
    ) -> str | Dict[str, Any]:
        """Parse once, then execute until completion or a user decision is needed."""
        query = deepcopy(query)
        system_prompt = (
            "You parse user instructions into file system tool calls. "
            "If the request can be divided into multiple operations, return one tool call "
            "for each operation in the required execution order. "
            "Do not omit operations from the user's request. "
            "For existing duplicate cleanup, use deduplicate_files. It handles selection and deletion "
            "after terminal confirmation; never add delete_file calls to implement its cleanup."
        )
        query.messages = [{"role": "system", "content": system_prompt}] + query.messages
        query.tools = storage_syscalls
        parser_response = self.execute_llm_syscall(agent_name, query)["response"]
        operations = deepcopy(parser_response.tool_calls)
        if not operations:
            return "No file operations were generated."
        if not isinstance(operations, list) or any(
            not isinstance(operation, dict)
            or not isinstance(operation.get("name"), str)
            or not isinstance(operation.get("parameters"), dict)
            for operation in operations
        ):
            raise ValueError("The LLM returned invalid file operations")

        state = FileOperationState(
            request_id=uuid4().hex,
            agent_name=agent_name,
            operations=operations,
        )
        return self._run_file_operations(state)

    def _get_confirmation_request(
        self, state: FileOperationState, operation: Dict[str, Any]
    ) -> Optional[str]:
        """Return a question when a policy requires input; no policy is connected yet."""
        return None

    def _pause_file_operation(
        self, state: FileOperationState, message: str, *, deduplication: Optional[dict] = None
    ) -> Dict[str, Any]:
        operation = state.operations[state.next_index]
        parameters = operation["parameters"]
        preview = {
            "name": operation["name"],
            "file_path": parameters.get("file_path", parameters.get("dir_path")),
        }
        content = parameters.get("content")
        if isinstance(content, str):
            preview.update({
                "content_preview": content[:1000],
                "content_truncated": len(content) > 1000,
                "size_bytes": len(content.encode("utf-8")),
            })
        with self.pending_operations_lock:
            state.confirmation_id = uuid4().hex
            state.status = "needs_confirmation"
            self.pending_operations[state.request_id] = state
            return {
                "status": state.status,
                "request_id": state.request_id,
                "confirmation_id": state.confirmation_id,
                "message": message,
                "operation": preview,
                "completed_count": state.next_index,
                "total_count": len(state.operations),
                **({"deduplication": deduplication} if deduplication is not None else {}),
            }

    def _pause_deduplication(self, state):
        def preview(record):
            return {key: record[key] for key in
                    ("id", "file_path", "size_bytes", "modified_at", "preview")}
        scan = state.deduplication_scan
        details = {"root_dir": scan["root_dir"], "threshold": scan["threshold"],
                   "scanned_count": scan["scanned_count"], "skipped": scan["skipped"]}
        if state.deletion_plan is None:
            details.update(stage="select", groups=[
                {"id": group["id"], "files": [preview(item) for item in group["files"]],
                 "pairs": group["pairs"]} for group in scan["groups"]
            ])
            message = "Select files to delete in each group. No files have been deleted."
        else:
            plan = state.deletion_plan
            details.update(stage="delete", retained=[preview(item) for item in plan["retained"]],
                           deletions=[{"file": preview(item["file"])} for item in plan["deletions"]])
            message = "Confirm the deletion list."
        return self._pause_file_operation(state, message, deduplication=details)

    def _run_file_operations(
        self, state: FileOperationState, *, approved_index: Optional[int] = None
    ) -> str | Dict[str, Any]:
        paused = False
        try:
            while state.next_index < len(state.operations):
                operation = state.operations[state.next_index]
                if operation["name"] == "deduplicate_files":
                    if state.deletion_plan is None:
                        scan_response = self.execute_storage_syscall(state.agent_name, StorageQuery(
                            operation_type="deduplicate_files", params={}
                        ))["response"]
                        scan = getattr(scan_response, "deduplication", None)
                        if scan is None:
                            raise ValueError(scan_response.response_message)
                        if scan["groups"]:
                            state.deduplication_scan = scan
                            response = self._pause_deduplication(state)
                            paused = True
                            return response
                        result = (f"No duplicate candidates found (cosine >= {scan['threshold']}); "
                                  f"compared {scan['scanned_count']} files. "
                                  f"Skipped files: {json.dumps(scan['skipped'], ensure_ascii=False)}")
                    else:
                        if state.next_index != approved_index:
                            raise ValueError("Duplicate deletion requires confirmation")
                        response = self.execute_storage_syscall(state.agent_name, ApplyDeduplicationQuery(
                            operation_type="deduplicate_files", params={}, plan=state.deletion_plan
                        ))["response"]
                        result = response.response_message
                        if response.error:
                            raise ValueError(result)
                    state.results.append({"operation": operation["name"], "parameters": {}, "result": result})
                    state.deduplication_scan = None
                    state.deletion_plan = None
                    state.next_index += 1
                    continue
                # Approval applies only to this operation, never subsequent writes.
                if state.next_index != approved_index:
                    message = self._get_confirmation_request(state, operation)
                    if message is not None:
                        response = self._pause_file_operation(state, message)
                        paused = True
                        return response
                storage_query = StorageQuery(
                    operation_type=operation["name"],
                    params=deepcopy(operation["parameters"]),
                )
                storage_response = self.execute_storage_syscall(state.agent_name, storage_query)
                response = storage_response.get("response")
                state.results.append({
                    "operation": operation["name"],
                    "parameters": deepcopy(operation["parameters"]),
                    "result": response.response_message if response is not None else "Storage operation failed",
                })
                state.next_index += 1
            state.status = "completed"
            return self._summarize_file_operations(state)
        except Exception as error:
            # Never replay earlier side effects after an execution failure.
            state.status = "failed"
            return {
                "status": state.status,
                "request_id": state.request_id,
                "message": f"File operation stopped: {error}",
                "completed_count": state.next_index,
                "results": state.results,
            }
        finally:
            if not paused:
                with self.pending_operations_lock:
                    self.pending_operations.pop(state.request_id, None)

    def resume_file_operation(
        self, request_id: str, confirmation_id: str, decision: str,
        delete_ids: Optional[Dict[str, List[str]]] = None,
    ) -> str | Dict[str, Any]:
        if decision not in ("approve", "cancel", "select"):
            raise OperationDecisionError("Decision must be approve, cancel, or select", 400)
        with self.pending_operations_lock:
            state = self.pending_operations.get(request_id)
            if state is None:
                raise OperationDecisionError("Pending operation not found or already finished", 404)
            if state.status != "needs_confirmation" or state.confirmation_id != confirmation_id:
                raise OperationDecisionError("This confirmation is no longer awaiting a decision")
            selecting = state.deduplication_scan is not None and state.deletion_plan is None
            if decision == "select":
                if not selecting:
                    raise OperationDecisionError("This operation is not awaiting a file selection", 400)
                try:
                    plan = build_deletion_plan(state.deduplication_scan, delete_ids or {})
                except (ValueError, TypeError) as error:
                    raise OperationDecisionError(str(error), 400) from error
                state.deletion_plan = plan
            elif decision == "approve" and selecting:
                raise OperationDecisionError("Select files to delete before approving deletion", 400)
            elif delete_ids is not None:
                raise OperationDecisionError("File IDs to delete are only accepted for selection", 400)
            # Claim this decision before releasing the lock or executing any syscall.
            state.confirmation_id = None
            if decision == "cancel":
                state.status = "cancelled"
                self.pending_operations.pop(request_id)
                return {
                    "status": state.status,
                    "request_id": request_id,
                    "message": "Remaining operations cancelled. Completed operations are unchanged.",
                    "completed_count": state.next_index,
                    "results": state.results,
                }
            state.status = "running"
            approved_index = state.next_index
        if decision == "select":
            return self._pause_deduplication(state)
        return self._run_file_operations(state, approved_index=approved_index)

    def _summarize_file_operations(self, state: FileOperationState) -> str:
        final_query = LLMQuery(
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the file operation results in execution order. "
                               "Try to be concise and maintain the key information including file name, file path, etc. "
                               "Write one concise line for each operation. "
                               "Preserve file paths and success or failure status exactly. "
                               "Do not provide alternatives, headings, emojis, or invented details."
                },
                {"role": "user", "content": f"File operation results: {json.dumps(state.results)}"},
            ],
            action_type="chat",
        )
        try:
            return self.execute_llm_syscall(state.agent_name, final_query)["response"].response_message
        except Exception:
            # A summary failure must not make completed writes look retryable.
            return "\n".join(str(result["result"]) for result in state.results)

    def execute_request(self, agent_name: str, query: Any) -> str | Dict[str, Any]:
        """
        Execute a request based on its type.
        
        Args:
            agent_name: Name of the agent making the request
            query: Query object of various types (LLM, Tool, Memory, Storage)
            
        Returns:
            Dict containing response and timing metrics
            
        Example:
            ```python
            query = LLMQuery(messages=[{"role": "user", "content": "Hello"}], action_type="chat")
            result = executor.execute_request("agent_1", query)
            ```
        """
        if isinstance(query, LLMQuery):
            if query.action_type == "chat" or query.action_type == "chat_with_json_response":
                llm_response = self.execute_llm_syscall(agent_name, query)
                return llm_response
            
            elif query.action_type == "tool_use":
                llm_response = self.execute_llm_syscall(agent_name, query)["response"]
                # breakpoint()
                tool_query = ToolQuery(
                    tool_calls=llm_response.tool_calls,
                    # action_type="tool_use"
                )
                tool_response = self.execute_tool_syscall(agent_name, tool_query)
                # breakpoint()
                return tool_response
            
            elif query.action_type == "operate_file":
                return self.execute_file_operation(agent_name, query)
        elif isinstance(query, ToolQuery):
            return self.execute_tool_syscall(agent_name, query)
        elif isinstance(query, MemoryQuery):
            if query.action_type == "add_agentic_memory":
                metadata = self.execute_memory_content_analyze(agent_name, query)
                query.params.update(metadata)
                # retrieve the related memory and evolve the memory.
                query.action_type = "retrieve_memory"
                similar_memories = self.execute_memory_syscall(agent_name, query)
                query, similar_memories_evolved = self.execute_memory_evolve(query, similar_memories)
                # define a memory abstract in the memory layer.
                query.action_type = "add_memory"
                for memory in similar_memories_evolved:
                    updated_query = MemoryQuery()
                    updated_query.params = memory.return_params() # return a dict with full parameters, also with memory id for updated
                    updated_query.action_type = "update_memory"
                    self.execute_memory_syscall(agent_name, updated_query)
                return self.execute_memory_syscall(agent_name, query)
            elif query.action_type == "add_memory":
                return self.execute_memory_syscall(agent_name, query)
            elif query.action_type == "remove_memory":
                return self.execute_memory_syscall(agent_name, query)
            elif query.action_type == "update_memory":
                return self.execute_memory_syscall(agent_name, query)
            elif query.action_type == "retrieve_memory":
                return self.execute_memory_syscall(agent_name, query)
            elif query.action_type == "get_memory":
                return self.execute_memory_syscall(agent_name, query)
        elif isinstance(query, StorageQuery):
            return self.execute_storage_syscall(agent_name, query)

def create_syscall_executor():
    """
    Create and return a SyscallExecutor instance and its wrapper.
    
    Returns:
        Tuple of (execute_request function, SyscallWrapper class)
        
    Example:
        ```python
        executor, wrapper = create_syscall_executor()
        response = executor("agent_1", LLMQuery(...))
        ```
    """
    executor = SyscallExecutor()
    
    class SyscallWrapper:
        """Wrapper class providing direct access to syscall methods."""
        llm = executor.execute_llm_syscall
        storage = executor.execute_storage_syscall
        memory = executor.execute_memory_syscall
        tool = executor.execute_tool_syscall
        resume_file_operation = executor.resume_file_operation

    return executor.execute_request, SyscallWrapper

# Maintain backwards compatibility
useSysCall = create_syscall_executor

