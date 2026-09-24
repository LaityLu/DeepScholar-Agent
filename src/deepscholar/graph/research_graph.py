from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.types import Send

from deepscholar.agents.citation_verifier import (
    CitationVerifierAgent,
)
from deepscholar.agents.claim_generator import (
    ClaimGeneratorAgent,
)
from deepscholar.agents.critic import CriticAgent
from deepscholar.agents.planner import PlannerAgent
from deepscholar.agents.replanner import (
    ReplannerAgent,
)
from deepscholar.agents.report_writer import (
    ReportWriterAgent,
)
from deepscholar.graph.state import (
    ResearchState,
    ResearchTaskState,
)
from deepscholar.models.citation import (
    VerifiedClaim,
)
from deepscholar.services.evidence_processor import (
    EvidenceProcessor,
)
from deepscholar.services.research_worker import (
    ResearchWorker,
)


class ResearchGraph:

    def __init__(
        self,
        planner: PlannerAgent,
        worker: ResearchWorker,
        critic: CriticAgent,
        replanner: ReplannerAgent,
        evidence_processor: EvidenceProcessor,
        claim_generator: ClaimGeneratorAgent,
        citation_verifier: CitationVerifierAgent,
        report_writer: ReportWriterAgent,
    ):
        self.planner = planner
        self.worker = worker
        self.critic = critic
        self.replanner = replanner

        self.evidence_processor = (
            evidence_processor
        )

        self.claim_generator = (
            claim_generator
        )

        self.citation_verifier = (
            citation_verifier
        )

        self.report_writer = (
            report_writer
        )

    # ==================================================
    # Build Graph
    # ==================================================

    def build(
        self,
        checkpointer=None,
    ):
        builder = StateGraph(
            ResearchState
        )

        # ------------------------------------------------
        # Nodes
        # ------------------------------------------------

        builder.add_node(
            "planner",
            self._planner_node,
        )

        builder.add_node(
            "research",
            self._research_node,
        )

        builder.add_node(
            "critic",
            self._critic_node,
        )

        builder.add_node(
            "replanner",
            self._replanner_node,
        )

        # Replanner 后用于触发新一轮 Send。
        builder.add_node(
            "replan_dispatch",
            self._replan_dispatch_node,
        )

        builder.add_node(
            "evidence_processor",
            self._evidence_processor_node,
        )

        builder.add_node(
            "claim_generator",
            self._claim_generator_node,
        )

        builder.add_node(
            "citation_verifier",
            self._citation_verifier_node,
        )

        builder.add_node(
            "report_writer",
            self._report_writer_node,
        )

        # ------------------------------------------------
        # START → Planner
        # ------------------------------------------------

        builder.add_edge(
            START,
            "planner",
        )

        # ------------------------------------------------
        # Planner
        # ↓
        # Send(task_001)
        # Send(task_002)
        # ...
        # ------------------------------------------------

        builder.add_conditional_edges(
            "planner",
            self._dispatch_initial_research,
        )

        # ------------------------------------------------
        # Parallel Research
        #
        # 所有 Research 分支写入：
        #
        # worker_results
        # evidences
        #
        # 通过 reducer 聚合。
        # ------------------------------------------------

        builder.add_edge(
            "research",
            "critic",
        )

        # ------------------------------------------------
        # Critic
        # ------------------------------------------------

        builder.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "complete": (
                    "evidence_processor"
                ),
                "replan": "replanner",
                "budget_exhausted": (
                    "evidence_processor"
                ),
            },
        )

        # ------------------------------------------------
        # Replanner
        # ------------------------------------------------

        builder.add_conditional_edges(
            "replanner",
            self._route_after_replanner,
            {
                "dispatch": (
                    "replan_dispatch"
                ),
                "complete": (
                    "evidence_processor"
                ),
            },
        )

        # ------------------------------------------------
        # Replan Dispatch
        #
        # 新任务再次 fan-out。
        # ------------------------------------------------

        builder.add_conditional_edges(
            "replan_dispatch",
            self._dispatch_replanned_research,
        )

        # ------------------------------------------------
        # Final pipeline
        # ------------------------------------------------

        builder.add_edge(
            "evidence_processor",
            "claim_generator",
        )

        builder.add_edge(
            "claim_generator",
            "citation_verifier",
        )

        builder.add_edge(
            "citation_verifier",
            "report_writer",
        )

        builder.add_edge(
            "report_writer",
            END,
        )

        return builder.compile(
            checkpointer=checkpointer
        )

    # ==================================================
    # Planner
    # ==================================================

    def _planner_node(
        self,
        state: ResearchState,
    ) -> dict:

        print(
            "\n[Graph] Enter Planner"
        )

        plan = self.planner.plan(
            state["user_query"]
        )

        return {
            "plan": plan,
        }

    # ==================================================
    # Initial Parallel Dispatch
    # ==================================================

    @staticmethod
    def _dispatch_initial_research(
        state: ResearchState,
    ) -> list[Send]:

        plan = state["plan"]

        if plan is None:
            raise RuntimeError(
                "Research plan is missing."
            )

        if not plan.tasks:
            raise RuntimeError(
                "Research plan contains no tasks."
            )

        print(
            f"\n[Graph] Dispatching "
            f"{len(plan.tasks)} initial "
            f"research tasks"
        )

        return [
            Send(
                "research",
                {
                    "task": task,
                },
            )
            for task in plan.tasks
        ]

    # ==================================================
    # Parallel Research Worker
    # ==================================================

    def _research_node(
        self,
        state: ResearchTaskState,
    ) -> dict:

        task = state["task"]

        print(
            f"\n[Research] START "
            f"{task.id}: {task.title}"
        )

        result = self.worker.run(
            task
        )

        if result.error:
            print(
                f"\n[Research] FAILED "
                f"{task.id}: {result.error}"
            )
        else:
            print(
                f"\n[Research] DONE "
                f"{task.id} "
                f"Evidence={len(result.evidences)}"
            )

        return {
            "worker_results": [
                result
            ],
            "evidences": (
                result.evidences
            ),
        }

    # ==================================================
    # Critic
    # ==================================================

    def _critic_node(
        self,
        state: ResearchState,
    ) -> dict:

        print(
            "\n[Graph] Enter Critic"
        )

        plan = state["plan"]

        if plan is None:
            raise RuntimeError(
                "Research plan is missing."
            )

        previous_critique = state[
            "previous_critique"
        ]

        # ------------------------------------------------
        # First global Critic
        # ------------------------------------------------

        if previous_critique is None:

            critique = self.critic.evaluate(
                plan=plan,
                evidences=state[
                    "evidences"
                ],
            )

        # ------------------------------------------------
        # Incremental Critic
        # ------------------------------------------------

        else:

            start_index = state[
                "last_critic_task_index"
            ]

            new_tasks = plan.tasks[
                start_index:
            ]

            new_task_ids = {
                task.id
                for task in new_tasks
            }

            new_evidences = [
                evidence
                for evidence
                in state["evidences"]
                if evidence.task_id
                in new_task_ids
            ]

            critique = (
                self.critic
                .evaluate_incremental(
                    plan=plan,
                    previous_critique=(
                        previous_critique
                    ),
                    new_tasks=new_tasks,
                    new_evidences=(
                        new_evidences
                    ),
                )
            )

        return {
            "critique": critique,

            "last_critic_task_index": (
                len(plan.tasks)
            ),
        }

    # ==================================================
    # Critic Router
    # ==================================================

    @staticmethod
    def _route_after_critic(
        state: ResearchState,
    ) -> str:

        critique = state[
            "critique"
        ]

        if critique is None:
            raise RuntimeError(
                "Critique is missing."
            )

        if critique.sufficient:
            return "complete"

        if (
            state["replan_count"]
            >= state["max_replans"]
        ):
            return "budget_exhausted"

        return "replan"

    # ==================================================
    # Replanner
    # ==================================================

    def _replanner_node(
        self,
        state: ResearchState,
    ) -> dict:

        print(
            "\n[Graph] Enter Replanner"
        )

        plan = state["plan"]

        critique = state[
            "critique"
        ]

        if plan is None:
            raise RuntimeError(
                "Research plan is missing."
            )

        if critique is None:
            raise RuntimeError(
                "Critique is missing."
            )

        result = self.replanner.replan(
            plan=plan,
            critique=critique,
        )

        next_replan_count = (
            state["replan_count"] + 1
        )

        # ------------------------------------------------
        # No new tasks
        # ------------------------------------------------

        if not result.new_tasks:

            return {
                "previous_critique": (
                    critique
                ),

                "has_new_tasks": False,

                "replan_count": (
                    next_replan_count
                ),
            }

        # ------------------------------------------------
        # Append new tasks
        # ------------------------------------------------

        new_plan = plan.model_copy(
            update={
                "tasks": (
                    plan.tasks
                    + result.new_tasks
                )
            }
        )

        return {
            "plan": new_plan,

            # Critic #N becomes compact memory
            # for Critic #(N+1).
            "previous_critique": (
                critique
            ),

            # Current critique becomes stale once
            # new research begins.
            "critique": None,

            "has_new_tasks": True,

            "replan_count": (
                next_replan_count
            ),
        }

    # ==================================================
    # Replanner Router
    # ==================================================

    @staticmethod
    def _route_after_replanner(
        state: ResearchState,
    ) -> str:

        if state["has_new_tasks"]:
            return "dispatch"

        return "complete"

    # ==================================================
    # Replan Dispatch Node
    # ==================================================

    @staticmethod
    def _replan_dispatch_node(
        state: ResearchState,
    ) -> dict:
        """
        No business logic.

        This node exists only to provide a clean
        graph boundary before dynamic Send fan-out.
        """

        return {}

    # ==================================================
    # Replanned Task Parallel Dispatch
    # ==================================================

    @staticmethod
    def _dispatch_replanned_research(
        state: ResearchState,
    ) -> list[Send]:

        plan = state["plan"]

        if plan is None:
            raise RuntimeError(
                "Research plan is missing."
            )

        start_index = state[
            "last_critic_task_index"
        ]

        new_tasks = plan.tasks[
            start_index:
        ]

        if not new_tasks:
            raise RuntimeError(
                "No replanned tasks available "
                "for dispatch."
            )

        print(
            f"\n[Graph] Dispatching "
            f"{len(new_tasks)} replanned "
            f"research tasks"
        )

        return [
            Send(
                "research",
                {
                    "task": task,
                },
            )
            for task in new_tasks
        ]

    # ==================================================
    # Evidence Processor
    # ==================================================

    def _evidence_processor_node(
        self,
        state: ResearchState,
    ) -> dict:

        print(
            "\n[Graph] Enter EvidenceProcessor"
        )

        processed_evidences = (
            self.evidence_processor.process(
                state["evidences"]
            )
        )

        if not processed_evidences:
            raise RuntimeError(
                "EvidenceProcessor produced "
                "no usable evidence."
            )

        return {
            "processed_evidences": (
                processed_evidences
            )
        }

    # ==================================================
    # Claim Generator
    # ==================================================

    def _claim_generator_node(
        self,
        state: ResearchState,
    ) -> dict:

        print(
            "\n[Graph] Enter ClaimGenerator"
        )

        plan = state["plan"]

        if plan is None:
            raise RuntimeError(
                "Research plan is missing."
            )

        processed_evidences = state[
            "processed_evidences"
        ]

        if not processed_evidences:
            raise RuntimeError(
                "Processed evidence pool is empty."
            )

        claims = (
            self.claim_generator.generate(
                goal=plan.goal,
                evidences=(
                    processed_evidences
                ),
            )
        )

        if not claims:
            raise RuntimeError(
                "ClaimGenerator produced no claims."
            )

        return {
            "claims": claims
        }

    # ==================================================
    # Citation Verifier
    # ==================================================

    def _citation_verifier_node(
        self,
        state: ResearchState,
    ) -> dict:

        print(
            "\n[Graph] Enter CitationVerifier"
        )

        claims = state[
            "claims"
        ]

        processed_evidences = state[
            "processed_evidences"
        ]

        if not claims:
            raise RuntimeError(
                "No claims are available "
                "for citation verification."
            )

        result = (
            self.citation_verifier.verify(
                claims=claims,
                evidences=(
                    processed_evidences
                ),
            )
        )

        claim_map = {
            claim.id: claim
            for claim in claims
        }

        verified_claims: list[
            VerifiedClaim
        ] = []

        for verification in (
            result.verifications
        ):
            if not verification.supported:
                continue

            if not (
                verification
                .supporting_evidence_ids
            ):
                continue

            claim = claim_map.get(
                verification.claim_id
            )

            if claim is None:
                continue

            verified_claims.append(
                VerifiedClaim(
                    id=claim.id,
                    text=claim.text,
                    section=claim.section,
                    supporting_evidence_ids=(
                        verification
                        .supporting_evidence_ids
                    ),
                )
            )

        if not verified_claims:
            raise RuntimeError(
                "CitationVerifier rejected "
                "all generated claims."
            )

        return {
            "citation_verification": (
                result
            ),

            "verified_claims": (
                verified_claims
            ),
        }

    # ==================================================
    # Report Writer
    # ==================================================

    def _report_writer_node(
        self,
        state: ResearchState,
    ) -> dict:

        print(
            "\n[Graph] Enter ReportWriter"
        )

        plan = state["plan"]

        if plan is None:
            raise RuntimeError(
                "Research plan is missing."
            )

        verified_claims = state[
            "verified_claims"
        ]

        if not verified_claims:
            raise RuntimeError(
                "No verified claims are available "
                "for report generation."
            )

        report = self.report_writer.write(
            goal=plan.goal,
            claims=verified_claims,
            evidences=state[
                "processed_evidences"
            ],
            critique=state[
                "critique"
            ],
        )

        if not report.strip():
            raise RuntimeError(
                "ReportWriter returned "
                "an empty report."
            )

        return {
            "report": report
        }