import { FolderOpen } from "lucide-react";
import { OpenPathToolbar } from "../common/OpenPathToolbar";
import { parentRunBrushLabelFromCheckpointPath } from "../../utils/training/checkpoints";
import {
  ENCODERS,
  EVAL_STRATEGIES,
  HPO_METHODS,
  META_STRATEGIES,
  MODELS,
  type TrainHubMode,
} from "./TrainHubSections";

function SelectField<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: readonly T[];
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-canvas-muted">{label}</label>
      <select
        className="select-base w-36"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

export function TrainHubModeSpecificCard({
  mode,
  model,
  setModel,
  encoder,
  setEncoder,
  batchSize,
  setBatchSize,
  epochs,
  setEpochs,
  hpoMethod,
  setHpoMethod,
  hpoTrials,
  setHpoTrials,
  hpoWorkers,
  setHpoWorkers,
  metaStrategy,
  setMetaStrategy,
  metaLr,
  setMetaLr,
  metaHistoryLength,
  setMetaHistoryLength,
  mrlBatchSize,
  setMrlBatchSize,
  mrlStep,
  setMrlStep,
  checkpointPath,
  setCheckpointPath,
  pickCheckpoint,
  evalDataset,
  setEvalDataset,
  pickDataset,
  evalStrategy,
  setEvalStrategy,
  evalSamples,
  setEvalSamples,
  projectRoot,
}: {
  mode: TrainHubMode;
  model: string;
  setModel: (v: string) => void;
  encoder: string;
  setEncoder: (v: string) => void;
  batchSize: number;
  setBatchSize: (v: number) => void;
  epochs: number;
  setEpochs: (v: number) => void;
  hpoMethod: string;
  setHpoMethod: (v: string) => void;
  hpoTrials: number;
  setHpoTrials: (v: number) => void;
  hpoWorkers: number;
  setHpoWorkers: (v: number) => void;
  metaStrategy: string;
  setMetaStrategy: (v: string) => void;
  metaLr: number;
  setMetaLr: (v: number) => void;
  metaHistoryLength: number;
  setMetaHistoryLength: (v: number) => void;
  mrlBatchSize: number;
  setMrlBatchSize: (v: number) => void;
  mrlStep: number;
  setMrlStep: (v: number) => void;
  checkpointPath: string;
  setCheckpointPath: (v: string) => void;
  pickCheckpoint: () => void;
  evalDataset: string;
  setEvalDataset: (v: string) => void;
  pickDataset: () => void;
  evalStrategy: string;
  setEvalStrategy: (v: string) => void;
  evalSamples: number;
  setEvalSamples: (v: number) => void;
  projectRoot: string | null;
}) {
  if (mode === "train" || mode === "hpo") {
    return (
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">
          {mode === "train" ? "Model Architecture" : "HPO Configuration"}
        </h2>
        <div className="flex flex-wrap gap-4 items-end">
          <SelectField
            label="Model"
            value={model}
            onChange={setModel}
            options={MODELS}
          />
          <SelectField
            label="Encoder"
            value={encoder}
            onChange={setEncoder}
            options={ENCODERS}
          />
          {mode === "train" && (
            <>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-canvas-muted">Batch Size</label>
                <input
                  type="number"
                  className="input-base font-mono text-sm w-24"
                  value={batchSize}
                  min={1}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-canvas-muted">Max Epochs</label>
                <input
                  type="number"
                  className="input-base font-mono text-sm w-24"
                  value={epochs}
                  min={1}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                />
              </div>
            </>
          )}
          {mode === "hpo" && (
            <>
              <SelectField
                label="Method"
                value={hpoMethod}
                onChange={setHpoMethod}
                options={HPO_METHODS}
              />
              <div className="flex flex-col gap-1">
                <label className="text-xs text-canvas-muted">Trials</label>
                <input
                  type="number"
                  className="input-base font-mono text-sm w-24"
                  value={hpoTrials}
                  min={1}
                  onChange={(e) => setHpoTrials(Number(e.target.value))}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-canvas-muted">Workers</label>
                <input
                  type="number"
                  className="input-base font-mono text-sm w-20"
                  value={hpoWorkers}
                  min={1}
                  onChange={(e) => setHpoWorkers(Number(e.target.value))}
                />
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  if (mode === "meta") {
    return (
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Meta-Learning</h2>
        <div className="flex flex-wrap gap-4 items-end">
          <SelectField
            label="Strategy"
            value={metaStrategy}
            onChange={setMetaStrategy}
            options={META_STRATEGIES}
          />
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">Meta LR</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-28"
              value={metaLr}
              min={0}
              step={0.000001}
              onChange={(e) => setMetaLr(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">History Length</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={metaHistoryLength}
              min={1}
              onChange={(e) => setMetaHistoryLength(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">Batch Size</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={mrlBatchSize}
              min={1}
              onChange={(e) => setMrlBatchSize(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">Meta Step</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-20"
              value={mrlStep}
              min={1}
              onChange={(e) => setMrlStep(Number(e.target.value))}
            />
          </div>
        </div>
        <p className="text-[10px] text-canvas-muted">
          Runs <code>main.py meta_train</code> with{" "}
          <code>envs@meta_rl.env</code> and <code>meta_rl.env.name</code> set
          to the problem selected above. The composed default in{" "}
          <code>meta_train.yaml</code> is <code>ptp</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="card space-y-4">
      <h2 className="text-sm font-semibold text-gray-200">Evaluation</h2>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-canvas-muted">Checkpoint</label>
        <div className="flex gap-2">
          <input
            type="text"
            className="input-base font-mono text-xs flex-1"
            value={checkpointPath}
            onChange={(e) => setCheckpointPath(e.target.value)}
            placeholder="path/to/best.pt"
          />
          <button
            onClick={pickCheckpoint}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <FolderOpen size={12} />
          </button>
        </div>
        {checkpointPath.trim() ? (
          <OpenPathToolbar
            path={checkpointPath.trim()}
            projectRoot={projectRoot}
            kind="checkpoint"
            labeled
            labeledIconSize={12}
            brushLabel={parentRunBrushLabelFromCheckpointPath(
              checkpointPath.trim(),
              projectRoot
            )}
            chipClassName="max-w-full"
          />
        ) : null}
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-canvas-muted">Dataset</label>
        <div className="flex gap-2">
          <input
            type="text"
            className="input-base font-mono text-xs flex-1"
            value={evalDataset}
            onChange={(e) => setEvalDataset(e.target.value)}
            placeholder="path/to/dataset.pkl"
          />
          <button
            onClick={pickDataset}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <FolderOpen size={12} />
          </button>
        </div>
        {evalDataset.trim() ? (
          <OpenPathToolbar
            path={evalDataset.trim()}
            projectRoot={projectRoot}
            chipClassName="max-w-full"
            handoff={false}
          />
        ) : null}
      </div>
      <div className="flex flex-wrap gap-4 items-end">
        <SelectField
          label="Strategy"
          value={evalStrategy}
          onChange={setEvalStrategy}
          options={EVAL_STRATEGIES}
        />
        <div className="flex flex-col gap-1">
          <label className="text-xs text-canvas-muted">Val Samples</label>
          <input
            type="number"
            className="input-base font-mono text-sm w-24"
            value={evalSamples}
            min={1}
            onChange={(e) => setEvalSamples(Number(e.target.value))}
          />
        </div>
      </div>
    </div>
  );
}
