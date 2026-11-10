"""GCP GKE Operations Agent with Google Kubernetes Engine, Observability, and GCP Knowledge MCP endpoints."""

import logging
import os
from typing import List, Optional

import google.auth
import google.auth.transport.requests
from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams,
)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

logger = logging.getLogger(__name__)

MODEL = os.getenv("ADK_MODEL", "gemini-2.5-flash")

# Managed Google Cloud MCP Endpoints (routed via Agent Egress Gateway)
GKE_MCP_URL = os.getenv("GKE_MCP_URL", "https://container.googleapis.com/mcp")
LOGGING_MCP_URL = os.getenv("LOGGING_MCP_URL", "https://logging.googleapis.com/mcp")
MONITORING_MCP_URL = os.getenv("MONITORING_MCP_URL", "https://monitoring.googleapis.com/mcp")
KNOWLEDGE_MCP_URL = os.getenv("KNOWLEDGE_MCP_URL", "https://developerknowledge.googleapis.com/mcp")


def _gcp_agent_identity_headers(_readonly_context=None) -> dict[str, str]:
    """Attach AGENT_IDENTITY OAuth Bearer token and resolved string project ID header to remote MCP calls."""
    headers: dict[str, str] = {}
    try:
        credentials, default_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        if credentials.token:
            headers["Authorization"] = f"Bearer {credentials.token}"
        raw_proj = (
            os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or default_project
            or "svcprjprd"
        )
        quota_project = "svcprjprd" if str(raw_proj).isdigit() else str(raw_proj)
        headers["X-Goog-User-Project"] = quota_project
    except Exception as exc:
        logger.warning("Failed to refresh AGENT_IDENTITY credentials for MCP call: %s", exc)
    return headers


class BestEffortMcpToolset(McpToolset):
    """Loads filtered MCP tools with AGENT_IDENTITY auth and keeps the agent running if egress denies a server."""

    def __init__(
        self,
        *,
        server_name: str,
        url: str,
        tool_name_prefix: str,
        tool_filter: Optional[List[str]] = None,
    ):
        self._server_name = server_name
        super().__init__(
            connection_params=StreamableHTTPConnectionParams(
                url=url,
                timeout=30.0,
            ),
            tool_filter=tool_filter,
            tool_name_prefix=tool_name_prefix,
            errlog=None,
            header_provider=_gcp_agent_identity_headers,
        )

    async def get_tools(self, readonly_context=None):
        try:
            return await super().get_tools(readonly_context)
        except Exception as exc:
            logger.warning("MCP server %s unavailable: %s", self._server_name, exc)
            return []


# ---------------------------------------------------------------------------
# 1. Google Kubernetes Engine (GKE) MCP (Read-Only Cluster/K8s Diagnostics Active)
# ---------------------------------------------------------------------------
gke_mcp_toolset = BestEffortMcpToolset(
    server_name="google-kubernetes-engine",
    url=GKE_MCP_URL,
    tool_name_prefix="gke_",
    tool_filter=[
        # --- ACTIVE: GKE Cluster, Node Pool & Kubernetes Workload Diagnostics ---
        "list_clusters",
        "get_cluster",
        "list_node_pools",
        "get_node_pool",
        "list_operations",
        "get_operation",
        "get_k8s_cluster_info",
        "get_k8s_cluster_health",
        "list_k8s_cluster_issues",
        "get_k8s_version",
        "list_k8s_api_resources",
        "check_k8s_auth",
        "describe_k8s_resource",
        "get_k8s_resource",
        "list_k8s_events",
        "get_k8s_rollout_status",
        "get_k8s_logs",
        "get_k8s_log_summary",
        "diagnose_k8s_workload",
        # --- FUTURE / DISABLED GKE MUTATING TOOLS (Uncomment to enable) ---
        # "create_cluster",
        # "update_cluster",
        # "create_node_pool",
        # "update_node_pool",
        # "cancel_operation",
        # "apply_k8s_manifest",
        # "delete_k8s_resource",
        # "patch_k8s_resource",
    ],
)

# ---------------------------------------------------------------------------
# 2. Google Cloud Observability MCPs (Cloud Logging + Cloud Monitoring)
# ---------------------------------------------------------------------------
logging_mcp_toolset = BestEffortMcpToolset(
    server_name="google-cloud-logging",
    url=LOGGING_MCP_URL,
    tool_name_prefix="logging_",
    tool_filter=[
        # --- ACTIVE: Read-Only Log Inspection ---
        "list_log_entries",
        "list_log_names",
        # --- FUTURE / DISABLED LOGGING TOOLS (Uncomment to enable) ---
        # "get_bucket",
        # "list_buckets",
        # "get_view",
        # "list_views",
    ],
)

monitoring_mcp_toolset = BestEffortMcpToolset(
    server_name="google-cloud-monitoring",
    url=MONITORING_MCP_URL,
    tool_name_prefix="monitoring_",
    tool_filter=[
        # --- ACTIVE: Read-Only Metrics & Alerts Inspection ---
        "list_timeseries",
        "query_range",
        "list_alert_policies",
        "list_alerts",
        # --- FUTURE / DISABLED MONITORING TOOLS (Uncomment to enable) ---
        # "get_alert_policy",
        # "get_alert",
        # "list_metric_descriptors",
        # "list_dashboards",
        # "get_dashboard",
    ],
)

# ---------------------------------------------------------------------------
# 3. Google Cloud Knowledge / Documentation MCP (Best Practices & Docs)
# ---------------------------------------------------------------------------
knowledge_mcp_toolset = BestEffortMcpToolset(
    server_name="google-developer-knowledge",
    url=KNOWLEDGE_MCP_URL,
    tool_name_prefix="knowledge_",
    tool_filter=[
        # --- ACTIVE: GCP Documentation & Best Practices Lookup ---
        "search_documents",
        "get_documents",
        # --- FUTURE / DISABLED KNOWLEDGE TOOLS (Uncomment to enable) ---
        # "answer_query",
    ],
)


root_agent = Agent(
    model=MODEL,
    name="gcp_gke_ops_agent",
    description=(
        "Google Kubernetes Engine (GKE) Operations Agent providing cluster, node pool, and workload "
        "diagnostics, Cloud Logging and Monitoring analysis, and official GCP documentation best practices."
    ),
    instruction=(
        "You are the Google Kubernetes Engine (GKE) Operations Specialist (`gcp_gke_ops_agent`).\n"
        "Follow an Observe -> Consult Best Practices -> Recommend workflow:\n"
        "1. OBSERVE: Use the GKE (`gke_*`), Cloud Logging (`logging_*`), and Cloud Monitoring "
        "(`monitoring_*`) tools to inspect GKE clusters, node pools, cluster health, Kubernetes resources, events, "
        "pod logs, rollout statuses, workload diagnostics, container metrics, and active alerts.\n"
        "2. CONSULT: Use the GCP Knowledge tools (`knowledge_*`) to look up official Google Kubernetes Engine documentation, "
        "security/autoscaling guides, and troubleshooting best practices.\n"
        "Never claim an action was executed unless a tool result proves it. If an MCP server is blocked by an "
        "Agent Egress Gateway policy, report which server was denied and continue with the available tools."
    ),
    tools=[
        gke_mcp_toolset,
        logging_mcp_toolset,
        monitoring_mcp_toolset,
        knowledge_mcp_toolset,
    ],
)
