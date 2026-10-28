import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AlgorithmVisualizer } from "../../src/pages/algorithms/visualizer/AlgorithmVisualizer";
import { useAppStore } from "../../src/store/app";
import { useAlgorithmDraftStore } from "../../src/store/algorithmDraft";
import type { ExecutionTraceDocument, TraceEvent } from "../../src/types/algorithms";
import fixture from "../fixtures/algorithm-trace.json";

const { listPolicyModules, tracePolicy } = vi.hoisted(() => ({
  listPolicyModules: vi.fn(),
  tracePolicy: vi.fn(),
}));

vi.mock("../../src/utils/algorithms/inspectClient", () => ({
  listPolicyModules,
  tracePolicy,
}));

const documentFixture = fixture as ExecutionTraceDocument;

function operatorEvent(): TraceEvent {
  const event = documentFixture.events.find((item) => item.name === "RandomRemoval");
  if (!event) throw new Error("fixture is missing RandomRemoval");
  return event;
}

function manyOperators(count: number): ExecutionTraceDocument {
  const events: TraceEvent[] = [
    {
      id: "e0",
      parent_id: null,
      name: "instance",
      kind: "input",
      changed_solution: false,
      arguments: {},
      before: null,
      after: null,
    },
    {
      id: "stage",
      parent_id: "e0",
      name: "alns",
      kind: "stage",
      changed_solution: false,
      arguments: {},
      before: null,
      after: null,
    },
  ];
  for (let index = 0; index < count; index += 1) {
    events.push({
      id: `op${index}`,
      parent_id: "stage",
      name: `Operator${index}`,
      kind: "operator",
      changed_solution: true,
      arguments: {},
      before: { routes: [[0, 1, 0]], objective: 1, load: null, feasible: true, unchanged_hash: "a" },
      after: { routes: [[0, 2, 0]], objective: 2, load: null, feasible: true, unchanged_hash: "b" },
    });
  }
  return {
    schema_version: 1,
    problem: "ptp",
    policies: { routing: "alns" },
    seed: 1,
    instance: {},
    events,
    final: { routes: [[0, 2, 0]], objective: 2, load: null, feasible: true, unchanged_hash: "b" },
  };
}

afterEach(() => {
  cleanup();
});

beforeEach(() => {
  listPolicyModules.mockReset();
  tracePolicy.mockReset();
  useAppStore.setState({ projectRoot: "" });
  useAlgorithmDraftStore.setState({
    problem: "ptp",
    graphSize: 5,
    seed: 1,
    packing: null,
    routing: "alns",
    model: null,
    routeImprovement: null,
    acceptance: null,
    operators: [],
  });
});

describe("AlgorithmVisualizer", () => {
  it("renders the fixture flow without calling the trace command", () => {
    render(<AlgorithmVisualizer initialDocument={documentFixture} />);
    expect(documentFixture.schema_version).toBe(1);
    const operator = operatorEvent();
    expect(operator.changed_solution).toBe(true);
    expect(operator.before?.routes).not.toEqual(operator.after?.routes);
    expect(screen.getAllByText("instance").length).toBeGreaterThan(0);
    expect(screen.getAllByText("RandomRemoval").length).toBeGreaterThan(0);
    expect(screen.getByTestId("final-route")).toHaveTextContent("0 → 1 → 2 → 0");
    expect(screen.getAllByText(/not_instrumented/).length).toBeGreaterThan(0);
    expect(screen.getByTestId("raw-stage-e1")).toHaveTextContent("hgs");
    expect(listPolicyModules).not.toHaveBeenCalled();
    expect(tracePolicy).not.toHaveBeenCalled();
  });

  it("moves the inspector with the arrow keys", () => {
    render(<AlgorithmVisualizer initialDocument={documentFixture} />);
    const flow = screen.getByRole("listbox", { name: "Execution flow" });
    expect(flow).toHaveAttribute("tabIndex", "0");
    fireEvent.keyDown(flow, { key: "ArrowDown" });
    fireEvent.keyDown(flow, { key: "ArrowDown" });
    fireEvent.keyDown(flow, { key: "ArrowDown" });
    expect(screen.getByTestId("inspector-title")).toHaveTextContent("RandomRemoval");
    expect(screen.getByTestId("before-routes")).toHaveTextContent("0 → 2 → 1 → 0");
    expect(screen.getByTestId("after-routes")).toHaveTextContent("0 → 1 → 2 → 0");
    fireEvent.keyDown(flow, { key: "ArrowUp" });
    expect(screen.getByTestId("inspector-title")).toHaveTextContent("alns");
  });

  it("collapses the middle of a long operator list", () => {
    render(<AlgorithmVisualizer initialDocument={manyOperators(45)} />);
    expect(screen.getByText("Operator0")).toBeInTheDocument();
    expect(screen.getByText("Operator19")).toBeInTheDocument();
    expect(screen.getByText("Operator35")).toBeInTheDocument();
    expect(screen.getByText("Operator44")).toBeInTheDocument();
    expect(screen.queryByText("Operator20")).not.toBeInTheDocument();
    expect(screen.queryByText("Operator34")).not.toBeInTheDocument();
    expect(screen.getByTestId("operator-gap")).toHaveTextContent("15 operator events hidden");
    fireEvent.click(screen.getByRole("button", { name: "Expand all" }));
    expect(screen.getByText("Operator20")).toBeInTheDocument();
    expect(screen.queryByTestId("operator-gap")).not.toBeInTheDocument();
  });

  it("does not trace a model id and keeps the loaded document", async () => {
    useAppStore.setState({ projectRoot: "/tmp/wsr" });
    useAlgorithmDraftStore.setState({ model: "am", routing: "alns" });
    listPolicyModules.mockResolvedValue({
      schema_version: 1,
      packing: [],
      routing: [
        { id: "alns", display_name: "alns", family: "routing", status: "registered", stages: [], consumes: [], produces: [] },
        { id: "hgs", display_name: "hgs", family: "routing", status: "present", stages: [], consumes: [], produces: [] },
      ],
      route_improvement: [],
      acceptance: [],
      operators: [],
      separation: [],
      models: [
        { id: "am", display_name: "am", family: "model", status: "shipped", role: "routing_model", stages: [], consumes: [], produces: [] },
        {
          id: "weight_adjustment_rnn",
          display_name: "WeightAdjustmentRNN",
          family: "model",
          status: "present",
          role: "reward_weight_adjustment",
          stages: [],
          consumes: [],
          produces: [],
        },
      ],
    });
    render(<AlgorithmVisualizer initialDocument={documentFixture} />);
    expect(await screen.findByTestId("algo-routing-status")).toHaveTextContent("registered");
    const routing = screen.getByLabelText("Routing");
    expect(routing.querySelector('option[value="alns"]')).toHaveClass("text-noc-cyan");
    expect(routing.querySelector('option[value="hgs"]')).toHaveClass("text-noc-textMuted");
    const models = screen.getByLabelText("Model");
    expect(models.querySelector('option[value="weight_adjustment_rnn"]')).toBeNull();
    expect(models.querySelector('option[value="am"]')).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Run trace" }));
    expect(screen.getByTestId("trace-run-state")).toHaveTextContent(
      "am is not runnable from the trace command yet."
    );
    expect(tracePolicy).not.toHaveBeenCalled();
    expect(screen.getByText("RandomRemoval")).toBeInTheDocument();
  });

  it("shows an empty catalog and a catalog command failure", async () => {
    useAppStore.setState({ projectRoot: "/tmp/wsr" });
    listPolicyModules.mockResolvedValueOnce({
      schema_version: 1,
      packing: [],
      routing: [],
      route_improvement: [],
      acceptance: [],
      operators: [],
      separation: [],
      models: [],
    });
    const { unmount } = render(<AlgorithmVisualizer />);
    expect(await screen.findByText("Policy catalog is empty.")).toBeInTheDocument();
    unmount();

    listPolicyModules.mockRejectedValueOnce(new Error("cli down"));
    render(<AlgorithmVisualizer />);
    expect(await screen.findByText(/Catalog command failed: cli down/)).toBeInTheDocument();
  });
});
