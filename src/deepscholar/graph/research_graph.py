from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from deepscholar.agents.citation_verifier import (
    CitationVerifierAgent,
)
from deepscholar.agents.claim_generator import (
    ClaimGeneratorAgent,
)
from deepscholar.agents.critic import CriticAgent
from deepscholar.agents.planner import PlannerAgent
from deepscholar.agents.report_writer import (
    ReportWriterAgent,
)
from deepscholar.agents.replanner import (
    ReplannerAgent,
)
from deepscholar.graph.state import ResearchState
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
    """
    Main DeepScholar research workflow.

    Workflow:

    Planner
        ↓
    Research
        ↓
    Critic
        ↓
    ┌──────── sufficient ──────────┐
    │                              │
    │                         EvidenceProcessor
    │                              ↓
    │                         ClaimGenerator
    │                              ↓
    │                       CitationVerifier
    │                              ↓
    │                         ReportWriter
    │                              ↓
    │                             END
    │
    └─ insufficient
           ↓
       Replanner
           ↓
       Research
           ↓
    Incremental Critic
           ↓
          ...
    """

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

    def build(self):
        """
        Build and compile the LangGraph workflow.
        """

        builder = StateGraph(
            ResearchState
        )

        # -------------------------
        # Nodes
        # -------------------------

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

        # -------------------------
        # Main research flow
        # -------------------------

        builder.add_edge(
            START,
            "planner",
        )

        builder.add_edge(
            "planner",
            "research",
        )

        # Run research tasks sequentially.
        builder.add_conditional_edges(
            "research",
            self._route_after_research,
            {
                "continue": "research",
                "critic": "critic",
            },
        )

        # Critic decides whether more research is needed.
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

        # Replanner may create incremental tasks.
        builder.add_conditional_edges(
            "replanner",
            self._route_after_replanner,
            {
                "research": "research",

                # If no useful new tasks can be produced,
                # continue with the best available evidence
                # rather than terminating without a report.
                "complete": (
                    "evidence_processor"
                ),
            },
        )

        # -------------------------
        # Report generation flow
        # -------------------------

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

        return builder.compile()

    # =====================================================
    # Planner
    # =====================================================

    def _planner_node(
        self,
        state: ResearchState,
    ) -> dict:
        """
        Generate the initial research plan.
        """

        plan = self.planner.plan(
            state["user_query"]
        )

        return {
            "plan": plan,
            "current_task_index": 0,
        }

    # =====================================================
    # Research
    # =====================================================

    def _research_node(
        self,
        state: ResearchState,
    ) -> dict:
        """
        Execute exactly one ResearchTask.

        The node intentionally does not hide a Python loop.
        One graph iteration corresponds to one research task.
        """

        plan = state["plan"]

        if plan is None:
            raise RuntimeError(
                "Research plan is missing."
            )

        index = state[
            "current_task_index"
        ]

        if index >= len(plan.tasks):
            raise RuntimeError(
                "Research task index is out of range."
            )

        task = plan.tasks[index]

        result = self.worker.run(
            task
        )

        return {
            "current_task_index": (
                index + 1
            ),
            "worker_results": [
                result
            ],
            "evidences": (
                result.evidences
            ),
        }

    @staticmethod
    def _route_after_research(
        state: ResearchState,
    ) -> str:
        """
        Continue executing tasks until all tasks in the
        current plan have been processed.
        """

        plan = state["plan"]

        if plan is None:
            raise RuntimeError(
                "Research plan is missing."
            )

        if (
            state["current_task_index"]
            < len(plan.tasks)
        ):
            return "continue"

        return "critic"

    # =====================================================
    # Critic
    # =====================================================

    def _critic_node(
        self,
        state: ResearchState,
    ) -> dict:
        """
        Run either:

        - global evaluation for the first Critic round
        - incremental evaluation after replanning
        """

        plan = state["plan"]

        if plan is None:
            raise RuntimeError(
                "Research plan is missing."
            )

        previous_critique = state[
            "previous_critique"
        ]

        # ---------------------------------------------
        # First-round global Critic
        # ---------------------------------------------

        if previous_critique is None:
            critique = self.critic.evaluate(
                plan=plan,
                evidences=state[
                    "evidences"
                ],
            )

        # ---------------------------------------------
        # Incremental Critic
        # ---------------------------------------------

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

            # Everything currently in the plan has now
            # been evaluated by the Critic.
            "last_critic_task_index": (
                len(plan.tasks)
            ),
        }

    @staticmethod
    def _route_after_critic(
        state: ResearchState,
    ) -> str:
        """
        Decide whether research is complete,
        should be replanned, or has exhausted
        its research budget.
        """

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

    # =====================================================
    # Replanner
    # =====================================================

    def _replanner_node(
        self,
        state: ResearchState,
    ) -> dict:
        """
        Generate incremental ResearchTasks from the
        current Critic knowledge gaps.
        """

        plan = state["plan"]
        critique = state["critique"]

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

        # ---------------------------------------------
        # No useful new tasks
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Append incremental tasks
        # ---------------------------------------------

        old_task_count = len(
            plan.tasks
        )

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

            # Begin research directly from the first
            # newly appended task.
            "current_task_index": (
                old_task_count
            ),

            # Save Critic #N so Critic #(N+1)
            # can perform incremental evaluation.
            "previous_critique": (
                critique
            ),

            # Current critique becomes stale after new
            # research starts.
            "critique": None,

            "has_new_tasks": True,

            "replan_count": (
                next_replan_count
            ),
        }

    @staticmethod
    def _route_after_replanner(
        state: ResearchState,
    ) -> str:
        """
        Continue researching when new tasks exist.

        If Replanner cannot generate useful tasks,
        proceed using the best evidence currently
        available.
        """

        if state["has_new_tasks"]:
            return "research"

        return "complete"

    # =====================================================
    # Evidence Processor
    # =====================================================

    def _evidence_processor_node(
        self,
        state: ResearchState,
    ) -> dict:
        """
        Deduplicate, quality-score, and curate the
        raw Evidence Pool.
        """

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

    # =====================================================
    # Claim Generator
    # =====================================================

    def _claim_generator_node(
        self,
        state: ResearchState,
    ) -> dict:
        """
        Generate atomic factual claims from curated
        evidence.
        """

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

    # =====================================================
    # Citation Verifier
    # =====================================================

    def _citation_verifier_node(
        self,
        state: ResearchState,
    ) -> dict:
        """
        Verify Claim -> Evidence support and construct
        VerifiedClaims using only citations accepted by
        the verifier.
        """

        claims = state["claims"]

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

    # =====================================================
    # Report Writer
    # =====================================================

    def _report_writer_node(
        self,
        state: ResearchState,
    ) -> dict:
        """
        Generate the final evidence-grounded report.
        """

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
            critique=state["critique"],
        )

        if not report.strip():
            raise RuntimeError(
                "ReportWriter returned "
                "an empty report."
            )

        return {
            "report": report
        }