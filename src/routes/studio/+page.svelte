<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import ProgressChart from '$lib/components/studio/ProgressChart.svelte';
  import TrainingGraph3D from '$lib/components/studio/TrainingGraph3D.svelte';

  // Use same origin when in browser (Vite proxy forwards /api to backend); env override for production
  const API = typeof window !== 'undefined' && (import.meta as any).env?.VITE_STUDIO_API
    ? (import.meta as any).env.VITE_STUDIO_API
    : typeof window !== 'undefined'
      ? ''  // browser: same origin → Vite dev proxy sends /api to localhost:8000
      : 'http://localhost:8000';  // SSR

  // ─── Dataset ───────────────────────────────────────────────────
  const DATASETS: Record<string, any> = {
    cifar10: {
      name: 'CIFAR-10', n_classes: 10, n_train: 50000, n_test: 10000, img_size: 32,
      classes: ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck'],
    },
    cifar100: {
      name: 'CIFAR-100', n_classes: 100, n_train: 50000, n_test: 10000, img_size: 32,
      classes: Array.from({length: 100}, (_, i) => `class_${i}`),
    },
  };

  let selectedDataset = 'cifar10';
  $: datasetInfo = DATASETS[selectedDataset];
  $: allClasses  = datasetInfo?.classes ?? [];

  // ─── Specialist buckets ────────────────────────────────────────
  interface Specialist {
    id: string;
    name: string;
    classes: number[];   // indices into allClasses
    color: string;
  }

  const COLORS = ['#d97706','#2563eb','#10b981','#7c3aed','#dc2626'];

  let specialists: Specialist[] = [
    { id: 'A', name: 'Specialist A', classes: [], color: COLORS[0] },
  ];
  let unassigned: number[] = [];   // class indices not in any specialist

  $: {
    const assigned = new Set(specialists.flatMap(s => s.classes));
    unassigned = allClasses.map((_: any, i: number) => i).filter((i: number) => !assigned.has(i));
  }

  function addSpecialist() {
    if (specialists.length >= 5) return;
    const id = String.fromCharCode(65 + specialists.length);
    specialists = [...specialists, { id, name: `Specialist ${id}`, classes: [], color: COLORS[specialists.length] }];
  }

  function removeSpecialist(idx: number) {
    specialists = specialists.filter((_, i) => i !== idx);
  }

  function moveClass(classIdx: number, fromSpec: number | null, toSpec: number | null) {
    specialists = specialists.map((s, i) => {
      if (i === fromSpec) return { ...s, classes: s.classes.filter(c => c !== classIdx) };
      if (i === toSpec)   return { ...s, classes: [...s.classes, classIdx].sort((a, b) => a - b) };
      return s;
    });
  }

  function moveUnassigned(classIdx: number, toSpec: number) {
    moveClass(classIdx, null, toSpec);
  }

  // drag-and-drop for class chips
  let draggingClass: number | null = null;
  let draggingFrom: number | null = null;   // null = unassigned pool

  function onChipDragStart(classIdx: number, fromSpec: number | null) {
    draggingClass = classIdx;
    draggingFrom = fromSpec;
  }

  function onBucketDrop(e: DragEvent, toSpec: number) {
    e.preventDefault();
    if (draggingClass === null) return;
    if (draggingFrom !== null) {
      moveClass(draggingClass, draggingFrom, toSpec);
    } else {
      moveUnassigned(draggingClass, toSpec);
    }
    draggingClass = null;
    draggingFrom = null;
  }

  function onPoolDrop(e: DragEvent) {
    e.preventDefault();
    if (draggingClass === null || draggingFrom === null) return;
    moveClass(draggingClass, draggingFrom, null);
    draggingClass = null;
    draggingFrom = null;
  }

  // ─── Canvas blocks ─────────────────────────────────────────────
  interface Block {
    id: string;
    type: 'specialist' | 'merge' | 'finetune' | 'note';
    specId?: string;
    x: number;
    y: number;
    status: 'idle' | 'training' | 'done' | 'error';
    valAcc?: number;
    testAcc?: number;
    mergeFrom?: string[];
    label?: string;   // for note blocks
  }

  let blocks: Block[] = [];
  let connections: { from: string; to: string }[] = [];

  // ── Selection (single + multi) ──────────────────────────────────
  let selectedIds: Set<string> = new Set();
  $: selectedBlockId = selectedIds.size === 1 ? [...selectedIds][0] : null;
  $: selectedBlock = selectedBlockId ? (blocks.find(b => b.id === selectedBlockId) ?? null) : null;
  $: selectedSpec  = selectedBlock?.type === 'specialist' ? specialists.find(s => s.id === selectedBlock?.specId) : null;
  $: selSpecCount  = blocks.filter(b => b.type === 'specialist' && selectedIds.has(b.id)).length;

  function selectOne(id: string) { selectedIds = new Set([id]); }
  function toggleSelect(id: string) {
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id); else next.add(id);
    selectedIds = next;
  }
  function clearSelection() { selectedIds = new Set(); }

  // Sync blocks from specialists
  $: {
    const existingIds = new Set(blocks.map(b => b.id));
    specialists.forEach((s, i) => {
      if (!existingIds.has(s.id)) {
        blocks = [...blocks, { id: s.id, type: 'specialist', specId: s.id, x: 60 + i * 220, y: 80, status: 'idle' }];
      }
    });
    blocks = blocks.filter(b => b.type !== 'specialist' || specialists.some(s => s.id === b.id));
  }

  // ── Block dragging (moves entire selection) ────────────────────
  let draggingBlock: string | null = null;
  let dragOffsetX = 0, dragOffsetY = 0;
  let dragStartPositions: Map<string, {x: number; y: number}> = new Map();
  let canvasEl: HTMLElement;

  // ── Rubber-band selection ──────────────────────────────────────
  let rubberBand: { x0: number; y0: number; x1: number; y1: number } | null = null;
  let isRubberBanding = false;
  // Flag: did the latest mousedown land on a block? prevents canvas from starting rubber-band
  let mousedownOnBlock = false;

  // ── Context menu ───────────────────────────────────────────────
  let ctxMenu: { x: number; y: number; blockId: string } | null = null;

  // ── Connect mode ───────────────────────────────────────────────
  let connectingFrom: string | null = null;
  let mousePos = { x: 0, y: 0 }; // live cursor for in-progress wire

  // ── Note editing ───────────────────────────────────────────────
  let editingNoteId: string | null = null;

  function onBlockMouseDown(e: MouseEvent, blockId: string) {
    // Right-clicks are handled by onBlockRightClick — don't interfere with ctxMenu
    if (e.button !== 0) return;
    // Do NOT stopPropagation — canvas mousedown will see mousedownOnBlock flag and skip rubber-band
    mousedownOnBlock = true;
    ctxMenu = null;
    if (connectingFrom) { finishConnect(blockId); return; }

    if (e.shiftKey) {
      toggleSelect(blockId);
    } else if (!selectedIds.has(blockId)) {
      selectOne(blockId);
    }
    // Begin drag — store start positions for all selected blocks
    draggingBlock = blockId;
    dragStartPositions = new Map(
      blocks.filter(b => selectedIds.has(b.id) || b.id === blockId).map(b => [b.id, { x: b.x, y: b.y }])
    );
    dragOffsetX = e.clientX;
    dragOffsetY = e.clientY;
  }

  function onBlockRightClick(e: MouseEvent, blockId: string) {
    e.preventDefault();
    e.stopPropagation();
    mousedownOnBlock = true;
    if (!selectedIds.has(blockId)) selectOne(blockId);
    const rect = canvasEl.getBoundingClientRect();
    ctxMenu = { x: e.clientX - rect.left, y: e.clientY - rect.top, blockId };
  }

  function onCanvasMouseDown(e: MouseEvent) {
    if (e.button !== 0) return;
    ctxMenu = null;
    if (connectingFrom) { connectingFrom = null; return; }
    // If mousedown was on a block, don't start rubber-band (block handler already ran)
    if (mousedownOnBlock) { mousedownOnBlock = false; return; }
    clearSelection();
    const rect = canvasEl.getBoundingClientRect();
    const x0 = e.clientX - rect.left;
    const y0 = e.clientY - rect.top;
    rubberBand = { x0, y0, x1: x0, y1: y0 };
    isRubberBanding = true;
  }

  function onCanvasMouseMove(e: MouseEvent) {
    const rect = canvasEl.getBoundingClientRect();
    mousePos = { x: e.clientX - rect.left, y: e.clientY - rect.top };

    if (draggingBlock) {
      const dx = e.clientX - dragOffsetX;
      const dy = e.clientY - dragOffsetY;
      blocks = blocks.map(b => {
        const start = dragStartPositions.get(b.id);
        if (!start) return b;
        return { ...b, x: Math.max(0, start.x + dx), y: Math.max(0, start.y + dy) };
      });
      return;
    }
    if (isRubberBanding && rubberBand) {
      rubberBand = { ...rubberBand, x1: e.clientX - rect.left, y1: e.clientY - rect.top };
    }
  }

  function onCanvasMouseUp(e: MouseEvent) {
    const wasRubberBanding = isRubberBanding;
    if (isRubberBanding && rubberBand) {
      const minX = Math.min(rubberBand.x0, rubberBand.x1);
      const maxX = Math.max(rubberBand.x0, rubberBand.x1);
      const minY = Math.min(rubberBand.y0, rubberBand.y1);
      const maxY = Math.max(rubberBand.y0, rubberBand.y1);
      if (maxX - minX > 4 || maxY - minY > 4) {
        selectedIds = new Set(
          blocks.filter(b => b.x + 104 > minX && b.x < maxX && b.y + 60 > minY && b.y < maxY).map(b => b.id)
        );
      }
      rubberBand = null;
    }
    isRubberBanding = false;
    draggingBlock = null;
    dragStartPositions = new Map();
    mousedownOnBlock = false;
  }

  function onCanvasClick(e: MouseEvent) {
    ctxMenu = null;
    // Only clear selection if this was a genuine empty-canvas click (not a block click or rubber-band end)
    if (!mousedownOnBlock && !isRubberBanding) clearSelection();
    mousedownOnBlock = false;
  }

  // ── Block operations ───────────────────────────────────────────
  function deleteBlock(blockId: string) {
    const block = blocks.find(b => b.id === blockId);
    if (!block) return;
    if (block.type === 'specialist') {
      const idx = specialists.findIndex(s => s.id === block.specId);
      if (idx !== -1) removeSpecialist(idx);
    } else {
      blocks = blocks.filter(b => b.id !== blockId);
    }
    connections = connections.filter(c => c.from !== blockId && c.to !== blockId);
    selectedIds.delete(blockId);
    selectedIds = new Set(selectedIds);
    ctxMenu = null;
  }

  function deleteSelected() {
    [...selectedIds].forEach(id => deleteBlock(id));
    clearSelection();
  }

  function duplicateBlock(blockId: string) {
    const block = blocks.find(b => b.id === blockId);
    if (!block || block.type === 'specialist') return;
    const newId = `${block.type}_${Date.now()}`;
    blocks = [...blocks, { ...block, id: newId, x: block.x + 30, y: block.y + 30, status: 'idle', valAcc: undefined, testAcc: undefined }];
    selectOne(newId);
    ctxMenu = null;
  }

  function duplicateSelected() {
    const nonSpec = [...selectedIds].filter(id => blocks.find(b => b.id === id)?.type !== 'specialist');
    const newIds: string[] = [];
    nonSpec.forEach(id => {
      const block = blocks.find(b => b.id === id)!;
      const newId = `${block.type}_${Date.now()}_${Math.random().toString(36).slice(2,5)}`;
      blocks = [...blocks, { ...block, id: newId, x: block.x + 30, y: block.y + 30, status: 'idle', valAcc: undefined, testAcc: undefined }];
      newIds.push(newId);
    });
    selectedIds = new Set(newIds);
    ctxMenu = null;
  }

  function resetBlockStatus(blockId: string) {
    blocks = blocks.map(b => b.id === blockId ? { ...b, status: 'idle', valAcc: undefined, testAcc: undefined } : b);
    ctxMenu = null;
  }

  function disconnectBlock(blockId: string) {
    connections = connections.filter(c => c.from !== blockId && c.to !== blockId);
    ctxMenu = null;
  }

  // Connect selected blocks → new merge block
  function connectSelectedToMerge() {
    const sel = blocks.filter(b => selectedIds.has(b.id));
    if (sel.length < 2) return;
    const id = `merge_${Date.now()}`;
    const cx = sel.reduce((s, b) => s + b.x, 0) / sel.length;
    const my = Math.max(...sel.map(b => b.y)) + 200;
    blocks = [...blocks, { id, type: 'merge', x: cx, y: my, status: 'idle', mergeFrom: sel.filter(b => b.type === 'specialist').map(b => b.specId!) }];
    sel.forEach(b => { connections = [...connections, { from: b.id, to: id }]; });
    selectOne(id);
  }

  function alignSelectedH() {
    const sel = [...selectedIds];
    if (sel.length < 2) return;
    const avgY = blocks.filter(b => sel.includes(b.id)).reduce((s, b) => s + b.y, 0) / sel.length;
    blocks = blocks.map(b => sel.includes(b.id) ? { ...b, y: Math.round(avgY) } : b);
  }

  function alignSelectedV() {
    const sel = [...selectedIds];
    if (sel.length < 2) return;
    const minX = Math.min(...blocks.filter(b => sel.includes(b.id)).map(b => b.x));
    let cur = minX;
    const sorted = [...blocks.filter(b => sel.includes(b.id))].sort((a, b) => a.x - b.x);
    sorted.forEach(b => {
      blocks = blocks.map(bl => bl.id === b.id ? { ...bl, x: cur } : bl);
      cur += 220;
    });
  }

  function startConnect(blockId: string) {
    connectingFrom = blockId;
    ctxMenu = null;
  }

  function finishConnect(blockId: string) {
    if (connectingFrom && connectingFrom !== blockId) {
      const exists = connections.some(c => c.from === connectingFrom && c.to === blockId);
      if (!exists) connections = [...connections, { from: connectingFrom!, to: blockId }];
    }
    connectingFrom = null;
  }

  function updateNoteLabel(blockId: string, e: Event) {
    const v = (e.target as HTMLTextAreaElement).value;
    blocks = blocks.map(b => b.id === blockId ? { ...b, label: v } : b);
  }

  function addNoteBlock() {
    const id = `note_${Date.now()}`;
    blocks = [...blocks, { id, type: 'note', x: 300, y: 200, status: 'idle', label: 'Note…' }];
    selectOne(id);
    editingNoteId = id;
  }

  // Keyboard shortcuts
  function onCanvasKeyDown(e: KeyboardEvent) {
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (editingNoteId) return; // let textarea handle it
      deleteSelected();
    }
    if (e.key === 'Escape') {
      clearSelection();
      ctxMenu = null;
      connectingFrom = null;
      editingNoteId = null;
    }
    if ((e.key === 'd' || e.key === 'D') && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (selectedIds.size > 1) duplicateSelected();
      else if (selectedBlockId) duplicateBlock(selectedBlockId);
    }
    if ((e.key === 'a' || e.key === 'A') && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      selectedIds = new Set(blocks.map(b => b.id));
    }
  }

  function addMergeBlock() {
    const sel = blocks.filter(b => selectedIds.has(b.id));
    const specBlocks = sel.length >= 2 ? sel : blocks.filter(b => b.type === 'specialist');
    const id = `merge_${Date.now()}`;
    const cx = specBlocks.length > 0 ? specBlocks.reduce((s, b) => s + b.x, 0) / specBlocks.length : 200;
    const my = specBlocks.length > 0 ? Math.max(...specBlocks.map(b => b.y)) + 200 : 340;
    blocks = [...blocks, { id, type: 'merge', x: cx, y: my, status: 'idle', mergeFrom: specBlocks.filter(b => b.type === 'specialist').map(b => b.specId!) }];
    specBlocks.forEach(b => { connections = [...connections, { from: b.id, to: id }]; });
    selectOne(id);
  }

  function addFinetuneBlock() {
    const mergeBlock = selectedBlockId && blocks.find(b => b.id === selectedBlockId && b.type === 'merge')
      ? blocks.find(b => b.id === selectedBlockId)
      : blocks.find(b => b.type === 'merge');
    const id = `ft_${Date.now()}`;
    const x = mergeBlock ? mergeBlock.x + 20 : 220;
    const y = mergeBlock ? mergeBlock.y + 220 : 560;
    blocks = [...blocks, { id, type: 'finetune', x, y, status: 'idle' }];
    if (mergeBlock) connections = [...connections, { from: mergeBlock.id, to: id }];
    selectOne(id);
  }

  // ─── Arch & Train Config ───────────────────────────────────────
  let archCfg = { n_layer: 4, n_embd: 128, n_head: 4, mlp_internal_dim_multiplier: 16, patch_size: 4, top_k_fraction: 0.15, dropout: 0.1 };
  let trainCfg = { epochs: 10, batch_size: 32, learning_rate: 1e-4, warmup_steps: 200, weight_decay: 0.05, grad_clip: 1.0, optimizer: 'adamw', lr_schedule: 'cosine', validation_split: 0.2, aug_random_crop: true, aug_horizontal_flip: true, aug_color_jitter: false, aug_mixup: false, max_samples: 2000, graph_update_every_n_batches: 10 };
  let pocMode = false;
  let prevEpochs = 50;
  function togglePoC() {
    if (pocMode) {
      trainCfg.max_samples = 1000;
      prevEpochs = trainCfg.epochs;
      trainCfg.epochs = 5;
    } else {
      trainCfg.max_samples = 0;
      trainCfg.epochs = prevEpochs;
    }
  }

  const N_EMBD_OPTIONS = [64, 96, 128, 192, 256, 384, 512];
  $: validHeads = [1,2,3,4,6,8,12,16].filter(h => archCfg.n_embd % h === 0);

  function clampHead() {
    if (!validHeads.includes(archCfg.n_head)) {
      archCfg.n_head = validHeads[validHeads.length - 1] ?? 1;
    }
  }
  $: archCfg.n_embd, clampHead();

  let estimatedParams = 0;
  async function fetchParamCount() {
    try {
      const r = await fetch(`${API}/api/studio/param-count?n_layer=${archCfg.n_layer}&n_embd=${archCfg.n_embd}&n_head=${archCfg.n_head}&mlp_internal_dim_multiplier=${archCfg.mlp_internal_dim_multiplier}&num_classes=10`);
      const d = await r.json();
      estimatedParams = d.params_m;
    } catch { estimatedParams = 0; }
  }
  $: archCfg && fetchParamCount();

  function resetToDefaults() {
    archCfg  = { n_layer: 4, n_embd: 128, n_head: 4, mlp_internal_dim_multiplier: 16, patch_size: 4, top_k_fraction: 0.15, dropout: 0.1 };
    trainCfg = { epochs: 10, batch_size: 32, learning_rate: 1e-4, warmup_steps: 200, weight_decay: 0.05, grad_clip: 1.0, optimizer: 'adamw', lr_schedule: 'cosine', validation_split: 0.2, aug_random_crop: true, aug_horizontal_flip: true, aug_color_jitter: false, aug_mixup: false, max_samples: 2000, graph_update_every_n_batches: 10 };
    pocMode = false;
  }

  // Right panel tabs
  let rightTab: 'arch' | 'train' | 'stats' = 'arch';
  function setTab(t: string) { rightTab = t as 'arch' | 'train' | 'stats'; }
  function tabClass(t: string) { return rightTab === t ? 'text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground'; }

  // ─── Job / Training ────────────────────────────────────────────
  let jobId: string | null = null;
  let jobStatus = 'idle';
  let currentEpoch = 0, totalEpochs = 0;
  let currentTrainingSpecName: string | null = null;  // which specialist is currently training (others show "queued")
  let trainLosses: number[] = [], valAccs: number[] = [];
  let logLines: string[] = [];
  let results: Record<string, {val_acc?: number; test_acc?: number; path?: string; params?: number; arch?: any}> = {};
  function getVal(r: any, key: string): string {
    return r?.[key] != null ? Number(r[key]).toFixed(1) : '—';
  }
  let mergeJobResult: any = null;
  let etaSeconds = 0;
  // ─── 3D Graph ───────────────────────────────────────────────────
  let graphSnapshot: { nodes: any[]; links: any[] } | null = null;
  let graphEpoch = 0;
  let graphBatch = 0;
  let graphLoss = 0;
  let graphValAcc = 0;
  let showGraph = false;
  let graphEverShown = false;  // tracks if graph was ever shown so we can offer reopen
  let eventSource: EventSource | null = null;

  // ─── Inference ──────────────────────────────────────────────────
  let inferCheckpoint: string | null = null;
  let inferImageB64: string | null = null;
  let inferImagePreview: string | null = null;  // object URL for <img>
  let inferResult: { class_probs: number[]; class_names: string[]; predicted_class: number; graph_snapshot: any } | null = null;
  let inferLoading = false;
  let inferError: string | null = null;
  let inferMode = false;   // when true — 3D graph shows activations instead of weights

  // ─── Floating panel minimize state ─────────────────────────────
  let isLeftMinimized = false;
  let isRightMinimized = false;
  let isGraphMinimized = false;

  // Keep a live cache of job states for checkpoint listing
  let JOBS_CACHE: Record<string, any> = {};
  // Merged graph snapshots keyed by job ID — restored on mount and updated after merge
  let mergedSnapshots: Record<string, { nodes: any[]; links: any[] }> = {};

  // When user selects a merged checkpoint, auto-show its architecture snapshot in the graph
  $: if (inferCheckpoint) {
    for (const [jobId, snap] of Object.entries(mergedSnapshots)) {
      if (JOBS_CACHE[jobId]?.merged_path === inferCheckpoint) {
        graphSnapshot = snap;
        showGraph = true;
        graphEverShown = true;
        break;
      }
    }
  }

  // Collect all available checkpoints from all jobs (path must always be a string for the API)
  function toPath(p: any): string {
    return typeof p === 'string' ? p : (p && typeof p === 'object' && 'path' in p ? String((p as any).path) : '');
  }
  $: availableCheckpoints = Object.values(JOBS_CACHE).flatMap((j: any) => {
    const ckpts: { label: string; path: string }[] = [];
    Object.entries(j.specialist_paths ?? {}).forEach(([name, path]) => {
      const pathStr = toPath(path);
      if (pathStr) ckpts.push({ label: `${name} (specialist)`, path: pathStr });
    });
    if (j.merged_path)    ckpts.push({ label: 'Merged model',    path: toPath(j.merged_path) });
    if (j.finetuned_path) ckpts.push({ label: 'Fine-tuned model', path: toPath(j.finetuned_path) });
    return ckpts;
  });

  function arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    const chunkSize = 8192;
    let binary = '';
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
      binary += String.fromCharCode.apply(null, chunk as unknown as number[]);
    }
    return btoa(binary);
  }

  async function handleInferImageUpload(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    if (inferImagePreview) URL.revokeObjectURL(inferImagePreview);
    inferImagePreview = URL.createObjectURL(file);
    try {
      const buf = await file.arrayBuffer();
      inferImageB64 = arrayBufferToBase64(buf);
      inferResult = null;
      inferError = null;
    } catch (err: any) {
      inferError = err?.message ?? 'Failed to read image';
      inferImageB64 = null;
    } finally {
      input.value = ''; // allow re-uploading same file
    }
  }

  async function runInference() {
    const pathStr = typeof inferCheckpoint === 'string' ? inferCheckpoint : (inferCheckpoint && typeof inferCheckpoint === 'object' && 'path' in inferCheckpoint ? (inferCheckpoint as any).path : null);
    if (!pathStr || !inferImageB64) return;
    inferLoading = true;
    inferError = null;
    inferResult = null;
    const url = `${API}/api/studio/infer`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 min for model load + inference
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          checkpoint_path: pathStr,
          image_b64: inferImageB64,
          dataset: selectedDataset,
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!res.ok) { const t = await res.text(); throw new Error(t || `HTTP ${res.status}`); }
      inferResult = await res.json();
      // Switch 3D graph to show activations
      graphSnapshot = inferResult!.graph_snapshot;
      inferMode = true;
      showGraph = true;
      graphEverShown = true;
    } catch (err: any) {
      clearTimeout(timeoutId);
      const msg = err?.message ?? '';
      if (err?.name === 'AbortError') {
        inferError = 'Request timed out. Inference may be slow; try again.';
      } else if (msg === 'Failed to fetch' || msg.includes('Load failed') || msg.includes('NetworkError')) {
        inferError = `Cannot reach the backend. Is it running at ${API}? Start it with: python main.py (from the backend folder).`;
      } else {
        inferError = msg || 'Inference failed';
      }
    } finally {
      inferLoading = false;
    }
  }

  function exitInferMode() {
    inferMode = false;
    inferResult = null;
  }

  async function launchTraining() {
    if (specialists.every(s => s.classes.length === 0)) { alert('Assign classes to at least one specialist.'); return; }

    const specsPayload = specialists
      .filter(s => s.classes.length > 0)
      .map(s => ({
        name: s.name,
        target_classes: s.classes,
        arch: { ...archCfg },
        train: { ...trainCfg },
      }));

    try {
      const r = await fetch(`${API}/api/studio/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: selectedDataset, specialists: specsPayload }),
      });
      const d = await r.json();
      jobId = d.job_id;
      jobStatus = 'training';
      trainLosses = []; valAccs = []; logLines = [];
      rightTab = 'stats';

      // Mark specialist blocks as training (current specialist will be set on first progress)
      currentTrainingSpecName = null;
      blocks = blocks.map(b => b.type === 'specialist' ? { ...b, status: 'training' } : b);

      startSSE(d.job_id);
    } catch (e) {
      alert('Error starting training: ' + e);
    }
  }

  function startSSE(id: string) {
    if (eventSource) eventSource.close();
    eventSource = new EventSource(`${API}/api/studio/stream/${id}`);
    eventSource.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'log') {
        logLines = [...logLines.slice(-100), msg.message];
        // Set current specialist from log so progress bar shows active model even before first progress event
        const m = (msg.message || '').match(/^\[([^\]]+)\]\s+Starting on/);
        if (m && jobStatus === 'training') currentTrainingSpecName = m[1];
      } else if (msg.type === 'progress') {
        currentEpoch = msg.epoch;
        totalEpochs  = msg.total;
        trainLosses  = msg.train_losses ?? trainLosses;
        valAccs      = msg.val_accs ?? valAccs;
        etaSeconds   = msg.eta ?? 0;
        if (msg.specialist_name != null) currentTrainingSpecName = msg.specialist_name;
      } else if (msg.type === 'done') {
        results = msg.results ?? {};

        if (jobStatus === 'merging') {
          // Merge finished — mark merge block done, auto-add finetune block
          blocks = blocks.map(b => b.type === 'merge' ? { ...b, status: 'done' } : b);
          logLines = [...logLines, '✓ Merge complete.'];
          jobStatus = 'done';
          // Auto-add finetune block if not already present
          const hasFt = blocks.some(b => b.type === 'finetune');
          if (!hasFt) {
            const mergeBlock = blocks.find(b => b.type === 'merge');
            if (mergeBlock) {
              const ftId = `ft_${Date.now()}`;
              blocks = [...blocks, { id: ftId, type: 'finetune', x: mergeBlock.x + 20, y: mergeBlock.y + 220, status: 'idle' }];
              connections = [...connections, { from: mergeBlock.id, to: ftId }];
              logLines = [...logLines, '→ Fine-tune block added. Click ⚙ Fine-tune to continue.'];
            }
          }
        } else if (jobStatus === 'finetuning') {
          // Finetune finished — mark finetune block done
          blocks = blocks.map(b => b.type === 'finetune' ? { ...b, status: 'done', testAcc: msg.test_acc } : b);
          logLines = [...logLines, `✓ Fine-tune complete. Test acc: ${msg.test_acc?.toFixed(2) ?? '—'}%`];
          jobStatus = 'done';
        } else {
          currentTrainingSpecName = null;
          // Specialist training finished — mark all specialist blocks done
          blocks = blocks.map(b => {
            if (b.type === 'specialist') {
              const specName = specialists.find(s => s.id === b.specId)?.name;
              const res = specName ? results[specName] : null;
              return { ...b, status: 'done', valAcc: res?.val_acc, testAcc: res?.test_acc };
            }
            return b;
          });
          logLines = [...logLines, '✓ All training complete.'];
          jobStatus = 'done';

          // Auto-add merge block if 2+ specialists trained
          const doneSpeBlocks = blocks.filter(b => b.type === 'specialist' && b.status === 'done');
          const hasMerge = blocks.some(b => b.type === 'merge');
          if (doneSpeBlocks.length >= 2 && !hasMerge) {
            const cx = doneSpeBlocks.reduce((s, b) => s + b.x, 0) / doneSpeBlocks.length;
            const id = `merge_${Date.now()}`;
            blocks = [...blocks, { id, type: 'merge', x: cx, y: 340, status: 'idle', mergeFrom: doneSpeBlocks.map(b => b.specId!) }];
            doneSpeBlocks.forEach(b => { connections = [...connections, { from: b.id, to: id }]; });
            logLines = [...logLines, '→ Merge block added. Click ⊕ Merge to continue.'];
          }
        }
        // Update jobs cache so inference panel can pick up checkpoint paths (always name -> path string)
        const specPaths: Record<string, string> = {};
        if (msg.specialist_paths && typeof msg.specialist_paths === 'object') {
          Object.entries(msg.specialist_paths).forEach(([n, p]) => {
            specPaths[n] = typeof p === 'string' ? p : (p && typeof p === 'object' && 'path' in p ? (p as any).path : '');
          });
        }
        if (Object.keys(specPaths).length === 0 && results && typeof results === 'object') {
          Object.entries(results).forEach(([n, r]) => {
            const path = r && typeof r === 'object' && 'path' in r ? (r as any).path : null;
            if (path && typeof path === 'string') specPaths[n] = path;
          });
        }
        JOBS_CACHE[jobId!] = {
          specialist_paths: specPaths,
          merged_path: msg.merged_path ?? null,
          finetuned_path: msg.finetuned_path ?? null,
        };
        // Store merged graph snapshot if present (sent by backend after merge)
        if (msg.merged_snapshot && jobId) {
          mergedSnapshots[jobId] = msg.merged_snapshot;
          mergedSnapshots = { ...mergedSnapshots }; // trigger reactivity
          // Auto-display the merged architecture in the graph panel
          graphSnapshot = msg.merged_snapshot;
          showGraph = true;
          graphEverShown = true;
        }
        // Pre-select the most recent checkpoint for inference (paths must be strings)
        const allCkpts = Object.values(JOBS_CACHE).flatMap((j: any) => {
          const c: string[] = [];
          Object.values(j.specialist_paths ?? {}).forEach((p: any) => {
            if (typeof p === 'string') c.push(p); else if (p && typeof p === 'object' && typeof (p as any).path === 'string') c.push((p as any).path);
          });
          if (j.merged_path)    c.push(j.merged_path);
          if (j.finetuned_path) c.push(j.finetuned_path);
          return c;
        });
        if (allCkpts.length && !inferCheckpoint) inferCheckpoint = allCkpts[allCkpts.length - 1];
        eventSource?.close();
      } else if (msg.type === 'graph') {
        graphSnapshot = msg.snapshot;
        graphEpoch    = msg.epoch ?? graphEpoch;
        graphBatch    = msg.batch ?? 0;
        graphLoss     = msg.loss  ?? graphLoss;
        graphValAcc   = msg.val_acc ?? 0;
        if (msg.specialist_name != null && jobStatus === 'training') currentTrainingSpecName = msg.specialist_name;
        showGraph     = true;
        graphEverShown = true;
      } else if (msg.type === 'error') {
        jobStatus = 'error';
        logLines = [...logLines, `✗ Error: ${msg.message}`];
        // Mark whichever block type was active as errored
        blocks = blocks.map(b => {
          if (b.status === 'training') return { ...b, status: 'error' };
          return b;
        });
        eventSource?.close();
      } else if (msg.type === 'cancelled') {
        jobStatus = 'idle';
        logLines = [...logLines, `⏹ ${msg.message ?? 'Training cancelled'}`];
        blocks = blocks.map(b => b.status === 'training' ? { ...b, status: 'idle' } : b);
        eventSource?.close();
      }
    };
  }

  async function runMerge() {
    if (!jobId) { alert('No training job found.'); return; }

    // Use selected specialist blocks if 2+ are selected; otherwise fall back to all specialists with classes
    const selectedSpecBlocks = blocks.filter(b => b.type === 'specialist' && selectedIds.has(b.id));
    const specsToMerge = selectedSpecBlocks.length >= 2
      ? selectedSpecBlocks
          .map(b => specialists.find(s => s.id === b.specId))
          .filter((s): s is typeof specialists[0] => !!s && s.classes.length > 0)
      : specialists.filter(s => s.classes.length > 0);

    if (specsToMerge.length < 2) {
      alert('Select 2+ trained specialist blocks on the canvas (shift-click) before merging, or ensure at least 2 specialists have assigned classes.');
      return;
    }

    const specNames = specsToMerge.map(s => s.name);
    trainLosses = []; valAccs = []; logLines = [];
    rightTab = 'stats';

    const r = await fetch(`${API}/api/studio/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, specialist_names: specNames, finetune_after: false }),
    });
    const d = await r.json();
    currentTrainingSpecName = null;
    jobStatus = 'merging';
    startSSE(jobId);
    logLines = [...logLines, `→ Merging ${specNames.length} specialists: ${specNames.join(', ')}...`];

    // Update merge block status
    blocks = blocks.map(b => b.type === 'merge' ? { ...b, status: 'training' } : b);
  }

  async function runFinetune() {
    if (!jobId) { alert('No job found.'); return; }
    trainLosses = []; valAccs = []; logLines = [];
    rightTab = 'stats';

    const r = await fetch(`${API}/api/studio/finetune`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, epochs: Math.max(3, Math.floor(trainCfg.epochs * 0.3)), learning_rate: 5e-5, batch_size: trainCfg.batch_size, dataset: selectedDataset, graph_update_every_n_batches: trainCfg.graph_update_every_n_batches }),
    });
    currentTrainingSpecName = null;
    jobStatus = 'finetuning';
    startSSE(jobId);
    logLines = [...logLines, '→ Fine-tuning merged model...'];
    blocks = blocks.map(b => b.type === 'finetune' ? { ...b, status: 'training' } : b);
  }

  async function stopTraining() {
    if (!jobId) return;
    try {
      await fetch(`${API}/api/studio/cancel/${jobId}`, { method: 'POST' });
    } catch (_) {}
    jobStatus = 'idle';
    logLines = [...logLines, '⏹ Training cancelled.'];
    blocks = blocks.map(b => b.status === 'training' ? { ...b, status: 'idle' } : b);
    eventSource?.close();
  }

  function formatETA(s: number): string {
    if (s < 60) return `${Math.round(s)}s`;
    return `${Math.floor(s/60)}m ${Math.round(s%60)}s`;
  }

  // Restore checkpoints from backend on page load (JOBS_CACHE is in-memory only)
  onMount(async () => {
    try {
      const r = await fetch(`${API}/api/studio/jobs`);
      if (!r.ok) return;
      const { jobs } = await r.json();
      const newCache: Record<string, any> = {};
      const newSnapshots: Record<string, { nodes: any[]; links: any[] }> = {};
      for (const job of jobs) {
        const cr = await fetch(`${API}/api/studio/checkpoints/${job.job_id}`);
        if (!cr.ok) continue;
        const data = await cr.json();
        newCache[job.job_id] = {
          specialist_paths: data.specialists ?? {},
          merged_path: data.merged ?? null,
          finetuned_path: data.finetuned ?? null,
        };
        if (data.merged_snapshot) newSnapshots[job.job_id] = data.merged_snapshot;
      }
      JOBS_CACHE = newCache;
      mergedSnapshots = newSnapshots;
    } catch (e) {
      console.warn('Could not restore session checkpoints:', e);
    }
  });

  onDestroy(() => eventSource?.close());

  // Canvas SVG connections
  function getBlockCenter(b: Block) { return { x: b.x + 100, y: b.y + 60 }; }
</script>

<!-- ═══════════════════════════════════════════════════════════════ -->
<svelte:head><title>BDH Studio — No-Code Training Platform</title></svelte:head>

<div class="flex flex-col" style="height: calc(100vh - 56px);">

  <!-- ── Top toolbar ───────────────────────────────────────────── -->
  <div class="bg-surface/80 backdrop-blur-md border-b border-border/50 px-4 py-2 flex items-center gap-3 flex-shrink-0">
    <span class="font-serif text-foreground font-medium text-glow-cyan">BDH Studio</span>
    <span class="text-muted-foreground text-xs border-l border-border/50 pl-3">No-Code Training Platform</span>

    <div class="flex items-center gap-2 ml-auto">
      <div class="w-px h-5 bg-border/50 mx-1"/>

      <!-- Run pipeline -->
      {#if jobStatus === 'idle' || jobStatus === 'done' || jobStatus === 'error'}
        <button
          on:click={launchTraining}
          class="px-4 py-1.5 text-sm bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/60 text-amber-400 hover:text-amber-300 font-mono rounded transition-colors font-medium glow-amber"
        >▶ Run Pipeline</button>
      {:else}
        <div class="flex items-center gap-2">
          <span class="px-3 py-1.5 text-sm bg-amber-500/10 border border-amber-500/30 text-amber-500/60 font-mono rounded flex items-center gap-1.5">
            <span class="inline-block w-2 h-2 rounded-full bg-amber-500 animate-pulse"/>
            {jobStatus === 'merging' ? 'Merging...' : jobStatus === 'finetuning' ? 'Fine-tuning...' : 'Training...'}
          </span>
          <button
            on:click={stopTraining}
            class="px-3 py-1.5 text-sm bg-destructive/20 hover:bg-destructive/30 border border-destructive/60 text-destructive font-mono rounded transition-colors"
          >⏹ Stop</button>
        </div>
      {/if}

      <button
        on:click={runMerge}
        disabled={jobStatus === 'training' || jobStatus === 'merging' || jobStatus === 'finetuning'}
        title={selSpecCount >= 2 ? `Merge ${selSpecCount} selected specialists` : 'Shift-click 2+ specialist blocks to select for merge'}
        class="px-3 py-1.5 text-xs font-mono rounded transition-colors
          {jobStatus === 'merging' ? 'bg-primary/10 border border-primary/30 text-primary/50 cursor-not-allowed' : 'bg-primary/20 hover:bg-primary/30 border border-primary/60 text-primary hover:text-primary'}"
      >⊕ Merge{selSpecCount >= 2 ? ` (${selSpecCount})` : ''}</button>
      <button
        on:click={runFinetune}
        disabled={jobStatus === 'training' || jobStatus === 'merging' || jobStatus === 'finetuning'}
        class="px-3 py-1.5 text-xs font-mono rounded transition-colors
          {jobStatus === 'finetuning' ? 'bg-emerald-400/10 border border-emerald-400/30 text-emerald-400/50 cursor-not-allowed' : 'bg-emerald-400/20 hover:bg-emerald-400/30 border border-emerald-400/60 text-emerald-400'}"
      >⚙ Fine-tune</button>

      <!-- Reopen 3D graph button — shown when graph was closed but data still exists -->
      {#if graphEverShown && !showGraph}
        <div class="w-px h-5 bg-border/50 mx-1"/>
        <button
          on:click={() => { showGraph = true; isGraphMinimized = false; }}
          title="Reopen 3D weight graph"
          class="px-3 py-1.5 text-xs font-mono rounded transition-colors bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-400 flex items-center gap-1.5"
        >
          <span>⬡</span> 3D Graph
        </button>
      {/if}
    </div>
  </div>

  <!-- ── Canvas area (relative container for floating panels) ─────── -->
  <div class="relative flex-1 min-h-0 overflow-hidden">

    <!-- ════ LEFT PANEL — Dataset & Classes ════════════════════════ -->
    <div class="studio-panel-wrapper studio-panel-left" class:minimized={isLeftMinimized}>
      <div class="studio-panel-header">
        <span class="studio-panel-title">Dataset &amp; Classes</span>
        <button class="studio-panel-toggle" on:click={() => isLeftMinimized = !isLeftMinimized} aria-label="Toggle left panel">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {#if isLeftMinimized}
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
            {:else}
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
            {/if}
          </svg>
        </button>
      </div>
      <div class="studio-panel-content">
    <aside class="flex flex-col scrollbar-thin" style="width:288px">
      <div class="p-4 space-y-4">

        <!-- Section header -->
        <div>
          <h2 class="font-serif text-sm font-medium text-foreground">Dataset</h2>
          <p class="text-xs text-muted-foreground mt-0.5">Select and split classes into specialists</p>
        </div>

        <!-- Dataset selector -->
        <div class="space-y-1.5">
          <label class="text-xs font-mono text-muted-foreground uppercase tracking-wider">Source</label>
          <select bind:value={selectedDataset} class="w-full px-2 py-1.5 text-sm border border-border/60 rounded-lg bg-surface/80 font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60">
            {#each Object.entries(DATASETS) as [key, info]}
              <option value={key}>{info.name}</option>
            {/each}
          </select>
        </div>

        <!-- Dataset stats card -->
        <div class="bg-surface/40 border border-border/40 rounded-xl p-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs font-mono">
          <span class="text-muted-foreground">classes</span><span class="text-foreground font-medium">{datasetInfo.n_classes}</span>
          <span class="text-muted-foreground">train</span><span class="text-foreground font-medium">{datasetInfo.n_train.toLocaleString()}</span>
          <span class="text-muted-foreground">test</span><span class="text-foreground font-medium">{datasetInfo.n_test.toLocaleString()}</span>
          <span class="text-muted-foreground">img size</span><span class="text-foreground font-medium">{datasetInfo.img_size}×{datasetInfo.img_size}</span>
        </div>

        <div class="border-t border-border/40 pt-3"/>

        <!-- Unassigned pool -->
        {#if unassigned.length > 0}
          <div>
            <label class="text-xs font-mono text-muted-foreground uppercase tracking-wider">Unassigned Classes</label>
            <!-- svelte-ignore a11y-no-static-element-interactions -->
            <div
              class="mt-1.5 min-h-10 p-2 rounded-lg border border-dashed border-border/50 flex flex-wrap gap-1"
              on:dragover|preventDefault
              on:drop={onPoolDrop}
            >
              {#each unassigned as ci}
                <!-- svelte-ignore a11y-no-static-element-interactions -->
                <span
                  class="px-2 py-0.5 text-xs font-mono bg-secondary/60 text-muted-foreground rounded cursor-grab border border-border/30 hover:bg-secondary hover:text-foreground transition-colors"
                  draggable="true"
                  on:dragstart={() => onChipDragStart(ci, null)}
                >{allClasses[ci]}</span>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Specialist buckets -->
        <div class="space-y-3">
          {#each specialists as spec, si}
            <div>
              <div class="flex items-center justify-between mb-1">
                <div class="flex items-center gap-1.5">
                  <div class="w-2.5 h-2.5 rounded-full" style="background:{spec.color}"/>
                  <input
                    bind:value={spec.name}
                    class="text-xs font-mono text-foreground bg-transparent border-none outline-none w-28 hover:underline focus:underline"
                  />
                </div>
                {#if specialists.length > 1}
                  <button on:click={() => removeSpecialist(si)} class="text-muted-foreground/40 hover:text-destructive text-xs leading-none transition-colors">✕</button>
                {/if}
              </div>
              <!-- svelte-ignore a11y-no-static-element-interactions -->
              <div
                class="min-h-10 p-2 rounded-lg border flex flex-wrap gap-1 transition-colors"
                style="border-color: {spec.color}44; background: {spec.color}0d"
                on:dragover|preventDefault
                on:drop={(e) => onBucketDrop(e, si)}
              >
                {#each spec.classes as ci}
                  <!-- svelte-ignore a11y-no-static-element-interactions -->
                  <span
                    class="px-2 py-0.5 text-xs font-mono rounded cursor-grab border transition-colors"
                    style="background:{spec.color}22; color:{spec.color}; border-color:{spec.color}55"
                    draggable="true"
                    on:dragstart={() => onChipDragStart(ci, si)}
                  >{allClasses[ci]}</span>
                {/each}
                {#if spec.classes.length === 0}
                  <span class="text-xs text-muted-foreground/40 font-mono">drop classes here</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>

        {#if specialists.length < 5}
          <button on:click={addSpecialist} class="w-full py-1.5 text-xs font-mono text-muted-foreground border border-dashed border-border/50 rounded-lg hover:border-amber-500/60 hover:text-amber-400 transition-colors">
            + Add Specialist
          </button>
        {/if}

        <!-- Results table (post-training) -->
        {#if Object.keys(results).length > 0}
          <div class="border-t border-border/40 pt-3">
            <label class="text-xs font-mono text-muted-foreground uppercase tracking-wider">Results</label>
            <table class="w-full text-xs font-mono mt-2 border-collapse">
              <thead>
                <tr class="text-muted-foreground">
                  <th class="text-left pb-1">Model</th>
                  <th class="text-right pb-1">Val%</th>
                  <th class="text-right pb-1">Test%</th>
                </tr>
              </thead>
              <tbody>
                {#each Object.entries(results) as [name, r]}
                  <tr class="border-t border-border/30">
                    <td class="py-1 text-foreground truncate max-w-[80px]">{name}</td>
                    <td class="py-1 text-right text-primary">{getVal(r, 'val_acc')}</td>
                    <td class="py-1 text-right text-emerald-400">{getVal(r, 'test_acc')}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}

      </div>
    </aside>
      </div><!-- /studio-panel-content left -->
    </div><!-- /studio-panel-wrapper left -->
    {#if isLeftMinimized}
      <button class="studio-expand-btn studio-expand-left" on:click={() => isLeftMinimized = false} aria-label="Expand left panel">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
      </button>
    {/if}

    <!-- ════ CENTER CANVAS (fills the entire relative container) ═══ -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
    <main
      bind:this={canvasEl}
      class="absolute inset-0 overflow-hidden bg-background canvas-grid outline-none"
      style="z-index:0; cursor: {draggingBlock ? 'grabbing' : connectingFrom ? 'crosshair' : isRubberBanding ? 'crosshair' : 'default'}"
      tabindex="0"
      on:mousedown={onCanvasMouseDown}
      on:mousemove={onCanvasMouseMove}
      on:mouseup={onCanvasMouseUp}
      on:click={onCanvasClick}
      on:keydown={onCanvasKeyDown}
    >
      <!-- Block palette strip -->
      <div class="absolute top-3 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 glass-card bg-surface/80 backdrop-blur-md border border-border/50 rounded-xl px-2.5 py-1.5 shadow-lg flex-wrap max-w-[90%]">
        <!-- Always-visible: add blocks -->
        <span class="text-[10px] font-mono text-muted-foreground pr-1.5 border-r border-border/40 mr-0.5">add</span>
        <button on:click|stopPropagation={addMergeBlock}
          class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-primary/15 text-primary hover:bg-primary/25 border border-primary/40 transition-colors">⊕ Merge</button>
        <button on:click|stopPropagation={addFinetuneBlock}
          class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-emerald-400/15 text-emerald-400 hover:bg-emerald-400/25 border border-emerald-400/40 transition-colors">⚙ Fine-tune</button>
        <button on:click|stopPropagation={addNoteBlock}
          class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-amber-500/15 text-amber-400 hover:bg-amber-500/25 border border-amber-500/40 transition-colors">✎ Note</button>

        <!-- Single-select actions -->
        {#if selectedIds.size === 1 && selectedBlockId}
          <div class="w-px h-4 bg-border/40 mx-0.5"/>
          <button on:click|stopPropagation={() => selectedBlockId && startConnect(selectedBlockId)}
            class="px-2 py-0.5 text-[11px] font-mono rounded-lg border transition-colors
              {connectingFrom === selectedBlockId ? 'bg-amber-500/20 text-amber-400 border-amber-500/60' : 'bg-surface/60 text-muted-foreground hover:bg-surface-hover border-border/40 hover:text-foreground'}">⇢ Wire</button>
          {#if blocks.find(b => b.id === selectedBlockId)?.type !== 'specialist'}
            <button on:click|stopPropagation={() => selectedBlockId && duplicateBlock(selectedBlockId)}
              class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-surface/60 text-muted-foreground hover:bg-surface-hover hover:text-foreground border border-border/40 transition-colors" title="Ctrl+D">⧉ Dupe</button>
          {/if}
          <button on:click|stopPropagation={() => { if(selectedBlockId) { disconnectBlock(selectedBlockId); } }}
            class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-surface/60 text-muted-foreground hover:bg-surface-hover hover:text-foreground border border-border/40 transition-colors">⊘ Unwire</button>
          {#if blocks.find(b => b.id === selectedBlockId)?.status !== 'idle'}
            <button on:click|stopPropagation={() => selectedBlockId && resetBlockStatus(selectedBlockId)}
              class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-amber-500/15 text-amber-400 hover:bg-amber-500/25 border border-amber-500/40 transition-colors">↺ Reset</button>
          {/if}
          <button on:click|stopPropagation={() => selectedBlockId && deleteBlock(selectedBlockId)}
            class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-destructive/15 text-destructive hover:bg-destructive/25 border border-destructive/40 transition-colors" title="Del">✕ Del</button>
        {/if}

        <!-- Multi-select actions -->
        {#if selectedIds.size > 1}
          <div class="w-px h-4 bg-border/40 mx-0.5"/>
          <span class="text-[10px] font-mono text-muted-foreground">{selectedIds.size} selected</span>
          <button on:click|stopPropagation={connectSelectedToMerge}
            class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-primary/15 text-primary hover:bg-primary/25 border border-primary/40 transition-colors">⊕ Merge selected</button>
          <button on:click|stopPropagation={alignSelectedH}
            class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-surface/60 text-muted-foreground hover:bg-surface-hover hover:text-foreground border border-border/40 transition-colors" title="Align to same Y">― Align H</button>
          <button on:click|stopPropagation={alignSelectedV}
            class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-surface/60 text-muted-foreground hover:bg-surface-hover hover:text-foreground border border-border/40 transition-colors" title="Spread X evenly">| Align V</button>
          <button on:click|stopPropagation={duplicateSelected}
            class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-surface/60 text-muted-foreground hover:bg-surface-hover hover:text-foreground border border-border/40 transition-colors" title="Ctrl+D">⧉ Dupe all</button>
          <button on:click|stopPropagation={deleteSelected}
            class="px-2 py-0.5 text-[11px] font-mono rounded-lg bg-destructive/15 text-destructive hover:bg-destructive/25 border border-destructive/40 transition-colors" title="Del">✕ Del all</button>
        {/if}

        <!-- Connect mode indicator -->
        {#if connectingFrom}
          <div class="w-px h-4 bg-border/40 mx-0.5"/>
          <span class="text-[10px] font-mono text-primary animate-pulse">click target…</span>
          <button on:click|stopPropagation={() => connectingFrom = null} class="text-[10px] font-mono text-muted-foreground hover:text-foreground px-1">✕</button>
        {/if}
      </div>

      <!-- SVG: connections + live wire + rubber-band -->
      <svg class="absolute inset-0 w-full h-full pointer-events-none" style="z-index:0">
        {#each connections as conn}
          {#if blocks.find(b => b.id === conn.from) && blocks.find(b => b.id === conn.to)}
            {@const fromBlock = blocks.find(b => b.id === conn.from)}
            {@const toBlock   = blocks.find(b => b.id === conn.to)}
            {#if fromBlock && toBlock}
              {@const fx = fromBlock.x + 104}
              {@const fy = fromBlock.y + 60}
              {@const tx = toBlock.x + 104}
              {@const ty = toBlock.y + 60}
              {@const isSelected = selectedIds.has(conn.from) || selectedIds.has(conn.to)}
              <line x1={fx} y1={fy} x2={tx} y2={ty}
                stroke={isSelected ? 'hsl(40 100% 50%)' : 'hsl(240 20% 28%)'}
                stroke-width={isSelected ? 2 : 1.5}
                stroke-dasharray="5,3"
                marker-end="url(#arrow)"/>
            {/if}
          {/if}
        {/each}

        <!-- Live wire while connecting -->
        {#if connectingFrom}
          {@const fromBlock = blocks.find(b => b.id === connectingFrom)}
          {#if fromBlock}
            <line x1={fromBlock.x + 104} y1={fromBlock.y + 60} x2={mousePos.x} y2={mousePos.y}
              stroke="hsl(186 100% 50%)" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.8"/>
            <circle cx={mousePos.x} cy={mousePos.y} r="4" fill="hsl(186 100% 50%)" opacity="0.8"/>
          {/if}
        {/if}

        <!-- Rubber-band selection box -->
        {#if rubberBand && isRubberBanding}
          {@const rx = Math.min(rubberBand.x0, rubberBand.x1)}
          {@const ry = Math.min(rubberBand.y0, rubberBand.y1)}
          {@const rw = Math.abs(rubberBand.x1 - rubberBand.x0)}
          {@const rh = Math.abs(rubberBand.y1 - rubberBand.y0)}
          <rect x={rx} y={ry} width={rw} height={rh}
            fill="hsl(186 100% 50% / 0.06)" stroke="hsl(186 100% 50%)" stroke-width="1" stroke-dasharray="4,2"/>
        {/if}

        <defs>
          <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill="hsl(240 20% 35%)"/>
          </marker>
        </defs>
      </svg>

      <!-- Blocks -->
      {#each blocks as block (block.id)}
        {@const spec = specialists.find(s => s.id === block.specId)}
        {@const isSelected = selectedIds.has(block.id)}
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <div
          class="absolute select-none"
          style="left:{block.x}px; top:{block.y}px; z-index:{isSelected ? 10 : 1}; width:{block.type === 'note' ? '160px' : '208px'}; cursor: {connectingFrom && connectingFrom !== block.id ? 'crosshair' : 'grab'}"
          on:mousedown={(e) => { if (connectingFrom) { finishConnect(block.id); } else { onBlockMouseDown(e, block.id); } }}
          on:contextmenu={(e) => onBlockRightClick(e, block.id)}
        >
          <!-- Note block — minimal sticky-note style -->
          {#if block.type === 'note'}
            <div
              class="rounded-lg border transition-all bg-amber-500/10 backdrop-blur-sm
                {isSelected ? 'border-amber-400 ring-2' : 'border-amber-500/30'}"
              style={isSelected ? 'box-shadow: 0 0 0 2px hsl(40 100% 50% / 0.3)' : ''}
            >
              <div class="px-2 py-1.5 flex items-center justify-between border-b border-amber-500/20">
                <span class="text-[10px] font-mono text-amber-400">✎ note</span>
                {#if isSelected}
                  <button on:mousedown|stopPropagation on:click|stopPropagation={() => deleteBlock(block.id)}
                    class="text-[10px] text-amber-500/60 hover:text-destructive transition-colors">✕</button>
                {/if}
              </div>
              {#if editingNoteId === block.id}
                <!-- svelte-ignore a11y-autofocus -->
                <textarea
                  autofocus
                  class="w-full bg-transparent text-xs font-mono text-amber-300 p-2 resize-none outline-none rounded-b-lg"
                  rows="3"
                  value={block.label ?? ''}
                  on:mousedown|stopPropagation
                  on:input={(e) => updateNoteLabel(block.id, e)}
                  on:blur={() => editingNoteId = null}
                />
              {:else}
                <!-- svelte-ignore a11y-no-static-element-interactions -->
                <div class="px-2 py-2 text-xs font-mono text-amber-300/80 min-h-[48px] whitespace-pre-wrap break-words"
                  on:dblclick|stopPropagation={() => { editingNoteId = block.id; }}>
                  {block.label || 'double-click to edit'}
                </div>
              {/if}
            </div>

          {:else}
          <div
            class="glass-card bg-surface/70 backdrop-blur-sm rounded-lg border transition-all
              {isSelected ? 'border-amber-500/60 shadow-lg ring-2' : connectingFrom ? 'border-primary/40' : 'border-border/40'}"
            style={isSelected ? 'box-shadow: 0 0 0 2px hsl(40 100% 50% / 0.2)' : ''}
          >
            <!-- Block header -->
            <div class="px-2.5 py-2 border-b flex items-center justify-between rounded-t-lg"
              style="background: {block.type === 'specialist' ? (spec?.color ?? '#d97706') + '22' : block.type === 'merge' ? 'hsl(186 100% 50% / 0.12)' : 'hsl(150 100% 50% / 0.10)'}; border-color: {block.type === 'specialist' ? (spec?.color ?? '#d97706') + '44' : block.type === 'merge' ? 'hsl(186 100% 50% / 0.25)' : 'hsl(150 100% 50% / 0.20)'}"
            >
              <div class="flex items-center gap-1.5 min-w-0">
                <span class="text-xs flex-shrink-0" style="color:{block.type === 'specialist' ? (spec?.color ?? '#d97706') : block.type === 'merge' ? 'hsl(186 100% 60%)' : 'hsl(150 100% 60%)'}">
                  {block.type === 'specialist' ? '◆' : block.type === 'merge' ? '⊕' : '⚙'}
                </span>
                <span class="text-xs font-mono font-medium text-foreground truncate">
                  {block.type === 'specialist' ? (spec?.name ?? block.id) : block.type === 'merge' ? 'Merge' : 'Fine-tune'}
                </span>
              </div>
              <div class="flex items-center gap-1 flex-shrink-0">
                <!-- Status badge -->
                <span class="text-[10px] font-mono px-1.5 py-0.5 rounded-full
                  {block.status === 'idle' ? 'bg-muted text-muted-foreground' :
                   block.status === 'training' ? 'bg-amber-500/20 text-amber-400' :
                   block.status === 'done' ? 'bg-emerald-400/20 text-emerald-400' :
                   'bg-destructive/20 text-destructive'}"
                >
                  {#if block.status === 'training'}
                    <span class="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse mr-0.5"/>
                  {/if}
                  {block.status}
                </span>
                <!-- Delete button (shown on selected) -->
                {#if isSelected}
                  <button
                    on:mousedown|stopPropagation
                    on:click|stopPropagation={() => deleteBlock(block.id)}
                    class="w-4 h-4 flex items-center justify-center rounded text-muted-foreground/60 hover:text-destructive hover:bg-destructive/10 transition-colors text-[10px]"
                    title="Delete block (Del)"
                  >✕</button>
                {/if}
              </div>
            </div>

            <!-- Block body -->
            <div class="px-3 py-2 text-xs font-mono space-y-1.5">
              {#if block.type === 'specialist' && spec}
                <div class="text-muted-foreground leading-relaxed">
                  {spec.classes.slice(0, 4).map(ci => allClasses[ci]).join(', ')}{spec.classes.length > 4 ? ` +${spec.classes.length - 4}` : ''}
                </div>
                <div class="text-muted-foreground/60 text-[10px]">{spec.classes.length} classes</div>
              {:else if block.type === 'merge'}
                {@const selSpecNames = blocks
                  .filter(b2 => b2.type === 'specialist' && selectedIds.has(b2.id))
                  .map(b2 => specialists.find(s => s.id === b2.specId)?.name)
                  .filter(Boolean)}
                {#if selSpecNames.length >= 2}
                  <div class="text-[10px] font-mono text-primary/80">will merge: {selSpecNames.join(' + ')}</div>
                {:else}
                  <div class="text-muted-foreground/60 text-[10px]">shift-click 2+ specialists to select</div>
                {/if}
                <div class="text-muted-foreground/40 text-[10px]">neuron-dim concat (paper S7.1)</div>
              {:else}
                <div class="text-muted-foreground">Full-dataset fine-tune</div>
              {/if}

              {#if block.testAcc}
                <div class="mt-1 pt-1 border-t border-border/30">
                  <span class="text-muted-foreground">test acc </span>
                  <span class="text-emerald-400 font-medium">{block.testAcc.toFixed(1)}%</span>
                </div>
              {/if}

              {#if block.status === 'training'}
                {#if block.type === 'specialist'}
                  {@const specName = specialists.find(s => s.id === block.specId)?.name}
                  {@const isActive = specName === currentTrainingSpecName}
                  <div class="mt-1 pt-1 border-t border-border/30">
                    {#if isActive}
                      <div class="text-muted-foreground">{currentEpoch}/{totalEpochs} epochs</div>
                      <div class="w-full bg-border/40 rounded-full h-1 mt-1">
                        <div class="bg-amber-400 h-1 rounded-full transition-all" style="width:{totalEpochs > 0 ? (currentEpoch/totalEpochs*100) : 0}%"/>
                      </div>
                    {:else}
                      <div class="text-muted-foreground/60 text-[10px] font-mono">queued</div>
                    {/if}
                  </div>
                {:else if (block.type === 'merge' && jobStatus === 'merging') || (block.type === 'finetune' && jobStatus === 'finetuning')}
                  <div class="mt-1 pt-1 border-t border-border/30">
                    <div class="text-muted-foreground">{block.type === 'merge' ? 'Merging…' : `${currentEpoch}/${totalEpochs} epochs`}</div>
                    {#if block.type === 'finetune' && totalEpochs > 0}
                      <div class="w-full bg-border/40 rounded-full h-1 mt-1">
                        <div class="bg-amber-400 h-1 rounded-full transition-all" style="width:{currentEpoch/totalEpochs*100}%"/>
                      </div>
                    {/if}
                  </div>
                {/if}
              {/if}

              <!-- Per-block action buttons -->
              {#if block.type === 'merge' && block.status !== 'training'}
                <div class="mt-1.5 pt-1.5 border-t border-border/30">
                  <!-- svelte-ignore a11y-no-static-element-interactions -->
                  <button
                    on:mousedown|stopPropagation
                    on:click|stopPropagation={runMerge}
                    disabled={jobStatus === 'training' || jobStatus === 'merging' || jobStatus === 'finetuning'}
                    class="w-full text-[10px] font-mono py-1 rounded transition-colors
                      {block.status === 'done' ? 'bg-primary/15 text-primary border border-primary/40 hover:bg-primary/25' : 'bg-primary/20 text-primary border border-primary/50 hover:bg-primary/30'}
                      disabled:opacity-40 disabled:cursor-not-allowed"
                  >⊕ {block.status === 'done' ? 'Re-merge' : 'Run Merge'}</button>
                </div>
              {/if}
              {#if block.type === 'finetune' && block.status !== 'training'}
                <div class="mt-1.5 pt-1.5 border-t border-border/30">
                  <!-- svelte-ignore a11y-no-static-element-interactions -->
                  <button
                    on:mousedown|stopPropagation
                    on:click|stopPropagation={runFinetune}
                    disabled={jobStatus === 'training' || jobStatus === 'merging' || jobStatus === 'finetuning'}
                    class="w-full text-[10px] font-mono py-1 rounded transition-colors
                      {block.status === 'done' ? 'bg-emerald-400/15 text-emerald-400 border border-emerald-400/40 hover:bg-emerald-400/25' : 'bg-emerald-400/20 text-emerald-400 border border-emerald-400/50 hover:bg-emerald-400/30'}
                      disabled:opacity-40 disabled:cursor-not-allowed"
                  >⚙ {block.status === 'done' ? 'Re-finetune' : 'Run Fine-tune'}</button>
                </div>
              {/if}
            </div>
          </div>
          {/if}<!-- end {#if note} {:else} -->
        </div>
      {/each}

      <!-- Empty state -->
      {#if blocks.length === 0}
        <div class="absolute inset-0 flex flex-col items-center justify-center pointer-events-none" style="padding-top:60px">
          <p class="text-muted-foreground font-serif text-lg">Add specialists in the left panel</p>
          <p class="text-muted-foreground/50 text-xs mt-2 font-mono">use the palette above to add Merge / Fine-tune blocks</p>
          <p class="text-muted-foreground/30 text-xs mt-1 font-mono">right-click any block for options · Del to delete · Ctrl+D to duplicate</p>
        </div>
      {/if}

      <!-- Context menu -->
      {#if ctxMenu}
        {@const ctxBlock = blocks.find(b => b.id === ctxMenu?.blockId)}
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <div
          class="absolute z-50 glass-card bg-surface/90 backdrop-blur-md border border-border/60 rounded-lg shadow-xl py-1 min-w-[160px] font-mono text-xs"
          style="left:{ctxMenu.x}px; top:{ctxMenu.y}px"
          on:mousedown|stopPropagation
          on:click|stopPropagation
        >
          <div class="px-3 py-1.5 text-[10px] text-muted-foreground border-b border-border/40 mb-1 flex items-center gap-1.5">
            <span>{ctxBlock?.type === 'specialist' ? '◆' : ctxBlock?.type === 'merge' ? '⊕' : ctxBlock?.type === 'finetune' ? '⚙' : '✎'}</span>
            <span class="truncate">{ctxBlock?.type === 'specialist' ? (specialists.find(s => s.id === ctxBlock?.specId)?.name ?? ctxMenu.blockId) : ctxBlock?.type === 'merge' ? 'Merge block' : ctxBlock?.type === 'finetune' ? 'Fine-tune block' : 'Note'}</span>
            {#if selectedIds.size > 1}<span class="ml-auto text-amber-400">{selectedIds.size} sel</span>{/if}
          </div>
          {#if ctxBlock?.type !== 'note'}
            <button on:click={() => { if (ctxMenu) startConnect(ctxMenu.blockId); }}
              class="w-full text-left px-3 py-1.5 hover:bg-surface-hover text-foreground flex items-center gap-2 transition-colors">
              <span class="text-muted-foreground">⇢</span> Wire to…
            </button>
            <button on:click={() => { if (ctxMenu) disconnectBlock(ctxMenu.blockId); }}
              class="w-full text-left px-3 py-1.5 hover:bg-surface-hover text-muted-foreground flex items-center gap-2 transition-colors">
              <span>⊘</span> Remove wires
            </button>
          {/if}
          {#if selectedIds.size > 1}
            <button on:click={() => { connectSelectedToMerge(); ctxMenu = null; }}
              class="w-full text-left px-3 py-1.5 hover:bg-primary/10 text-primary flex items-center gap-2 transition-colors">
              <span>⊕</span> Merge selected ({selectedIds.size})
            </button>
            <button on:click={() => { alignSelectedH(); ctxMenu = null; }}
              class="w-full text-left px-3 py-1.5 hover:bg-surface-hover text-foreground flex items-center gap-2 transition-colors">
              <span>―</span> Align horizontal
            </button>
            <button on:click={() => { deleteSelected(); ctxMenu = null; }}
              class="w-full text-left px-3 py-1.5 hover:bg-destructive/10 text-destructive flex items-center gap-2 transition-colors">
              <span>✕</span> Delete selected
            </button>
          {:else}
            {#if ctxBlock?.type !== 'specialist' && ctxBlock?.type !== 'note'}
              <button on:click={() => { if (ctxMenu) duplicateBlock(ctxMenu.blockId); }}
                class="w-full text-left px-3 py-1.5 hover:bg-surface-hover text-foreground flex items-center gap-2 transition-colors">
                <span class="text-muted-foreground">⧉</span> Duplicate <span class="ml-auto text-muted-foreground/40">Ctrl+D</span>
              </button>
            {/if}
            {#if ctxBlock?.status !== 'idle' && ctxBlock?.type !== 'note'}
              <button on:click={() => { if (ctxMenu) resetBlockStatus(ctxMenu.blockId); }}
                class="w-full text-left px-3 py-1.5 hover:bg-amber-500/10 text-amber-400 flex items-center gap-2 transition-colors">
                <span>↺</span> Reset to idle
              </button>
            {/if}
            <div class="border-t border-border/40 mt-1"/>
            <button on:click={() => { if (ctxMenu) deleteBlock(ctxMenu.blockId); }}
              class="w-full text-left px-3 py-1.5 hover:bg-destructive/10 text-destructive flex items-center gap-2 transition-colors">
              <span>✕</span> Delete <span class="ml-auto text-muted-foreground/40">Del</span>
            </button>
          {/if}
        </div>
      {/if}
    </main>

    <!-- ════ RIGHT PANEL — Config & Stats ════════════════════════ -->
    <div class="studio-panel-wrapper studio-panel-right" class:minimized={isRightMinimized}>
      <div class="studio-panel-header">
        <span class="studio-panel-title">Config &amp; Stats</span>
        <button class="studio-panel-toggle" on:click={() => isRightMinimized = !isRightMinimized} aria-label="Toggle right panel">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {#if isRightMinimized}
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
            {:else}
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
            {/if}
          </svg>
        </button>
      </div>
      <div class="studio-panel-content">
    <aside class="flex flex-col" style="width:320px">

      <!-- Tabs -->
      <div class="flex border-b border-border/50 flex-shrink-0">
        {#each [['arch','Architecture'],['train','Training'],['stats','Live Stats']] as [tab, label]}
          <button
            on:click={() => setTab(tab)}
            class="flex-1 py-2 text-xs font-mono transition-colors {tabClass(tab)}"
          >{label}</button>
        {/each}
      </div>

      <div class="p-4 space-y-4 overflow-y-auto flex-1">

        <!-- ── ARCHITECTURE TAB ── -->
        {#if rightTab === 'arch'}
          <div class="flex items-center justify-between">
            <h3 class="font-serif text-sm text-foreground">Architecture Config</h3>
            <button on:click={resetToDefaults} class="text-xs font-mono text-amber-400 hover:text-amber-300 hover:underline transition-colors">reset defaults</button>
          </div>

          <!-- Param count badge -->
          <div class="bg-surface/40 border border-border/40 rounded-xl px-3 py-2 flex items-center justify-between">
            <span class="text-xs font-mono text-muted-foreground">est. parameters</span>
            <span class="text-sm font-mono font-bold text-amber-400">{estimatedParams ? `~${estimatedParams}M` : '...'}</span>
          </div>

          <div class="space-y-3">
            <div class="space-y-1">
              <div class="flex justify-between text-xs font-mono">
                <label class="text-muted-foreground">n_layer</label>
                <span class="text-foreground font-medium">{archCfg.n_layer}</span>
              </div>
              <input type="range" min="2" max="12" bind:value={archCfg.n_layer} class="w-full accent-cyan-400 h-1.5 rounded"/>
              <div class="flex justify-between text-[10px] font-mono text-muted-foreground/40"><span>2</span><span>12</span></div>
            </div>

            <div class="space-y-1">
              <label class="text-xs font-mono text-muted-foreground">n_embd</label>
              <select bind:value={archCfg.n_embd} class="w-full px-2 py-1 text-xs border border-border/60 rounded-lg bg-surface/60 font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60">
                {#each N_EMBD_OPTIONS as v}<option value={v}>{v}</option>{/each}
              </select>
            </div>

            <div class="space-y-1">
              <label class="text-xs font-mono text-muted-foreground">n_head</label>
              <select bind:value={archCfg.n_head} class="w-full px-2 py-1 text-xs border border-border/60 rounded-lg bg-surface/60 font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60">
                {#each validHeads as h}<option value={h}>{h}</option>{/each}
              </select>
              <p class="text-[10px] text-muted-foreground/40 font-mono">must divide n_embd ({archCfg.n_embd})</p>
            </div>

            <div class="space-y-1">
              <div class="flex justify-between text-xs font-mono">
                <label for="arch-mlp-mult" class="text-muted-foreground">mlp_multiplier</label>
                <span class="text-foreground font-medium">{archCfg.mlp_internal_dim_multiplier}</span>
              </div>
              <input id="arch-mlp-mult" type="range" min="4" max="64" step="4" bind:value={archCfg.mlp_internal_dim_multiplier} class="w-full accent-cyan-400 h-1.5 rounded"/>
              <div class="flex justify-between text-[10px] font-mono text-muted-foreground/40"><span>4</span><span>64</span></div>
            </div>

            <div class="space-y-1" role="group" aria-labelledby="arch-patch-label">
              <span id="arch-patch-label" class="text-xs font-mono text-muted-foreground">patch_size</span>
              <div class="flex gap-1">
                {#each [2,4,8] as ps}
                  <button
                    on:click={() => archCfg.patch_size = ps}
                    class="flex-1 py-1 text-xs font-mono border rounded-lg transition-colors
                      {archCfg.patch_size === ps ? 'bg-primary/20 border-primary/60 text-primary' : 'border-border/40 text-muted-foreground hover:bg-surface-hover'}"
                  >{ps}px</button>
                {/each}
              </div>
            </div>

            <div class="space-y-1">
              <div class="flex justify-between text-xs font-mono">
                <label for="arch-topk" class="text-muted-foreground">top_k_fraction <span class="text-muted-foreground/40">(sparsity)</span></label>
                <span class="text-foreground font-medium">{archCfg.top_k_fraction.toFixed(2)}</span>
              </div>
              <input id="arch-topk" type="range" min="0.05" max="0.30" step="0.01" bind:value={archCfg.top_k_fraction} class="w-full accent-cyan-400 h-1.5 rounded"/>
              <div class="flex justify-between text-[10px] font-mono text-muted-foreground/40"><span>5%</span><span>30%</span></div>
            </div>

            <div class="space-y-1">
              <div class="flex justify-between text-xs font-mono">
                <label for="arch-dropout" class="text-muted-foreground">dropout</label>
                <span class="text-foreground font-medium">{archCfg.dropout.toFixed(2)}</span>
              </div>
              <input id="arch-dropout" type="range" min="0" max="0.5" step="0.01" bind:value={archCfg.dropout} class="w-full accent-cyan-400 h-1.5 rounded"/>
            </div>
          </div>

        <!-- ── TRAINING TAB ── -->
        {:else if rightTab === 'train'}
          <h3 class="font-serif text-sm text-foreground">Training Config</h3>

          <!-- PoC Quick Train toggle -->
          <div class="p-2.5 rounded-lg border-2 {pocMode ? 'border-amber-500/60 bg-amber-500/10' : 'border-dashed border-border/50 bg-surface/40'}">
            <label class="flex items-center gap-2.5 cursor-pointer">
              <input type="checkbox" bind:checked={pocMode} on:change={togglePoC} class="accent-amber-400 w-4 h-4"/>
              <div>
                <div class="text-xs font-mono font-bold {pocMode ? 'text-amber-400' : 'text-muted-foreground'}">⚡ Quick Train (PoC)</div>
                <div class="text-[10px] font-mono {pocMode ? 'text-amber-400/70' : 'text-muted-foreground/60'}">
                  {pocMode ? 'dataset cached after first run' : 'fast demo with subset of data'}
                </div>
              </div>
            </label>
            {#if pocMode}
              <div class="mt-2 flex items-center gap-2">
                <label for="train-max-samples" class="text-[10px] font-mono text-amber-400 whitespace-nowrap">max samples</label>
                <input id="train-max-samples" type="number" min="100" max="10000" step="100"
                  bind:value={trainCfg.max_samples}
                  class="w-full text-xs font-mono bg-surface/60 border border-amber-500/40 rounded-lg px-2 py-0.5 text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-400/60"/>
              </div>
            {/if}
          </div>

          <div class="space-y-3">
            <div class="space-y-1">
              <div class="flex justify-between text-xs font-mono">
                <label class="text-muted-foreground">epochs</label>
                <span class="text-foreground font-medium">{trainCfg.epochs}</span>
              </div>
              <input type="range" min="5" max="200" bind:value={trainCfg.epochs} class="w-full accent-cyan-400 h-1.5 rounded"/>
              <div class="flex justify-between text-[10px] font-mono text-muted-foreground/40"><span>5</span><span>200</span></div>
            </div>

            <div class="space-y-1">
              <label class="text-xs font-mono text-muted-foreground">batch_size</label>
              <select bind:value={trainCfg.batch_size} class="w-full px-2 py-1 text-xs border border-border/60 rounded-lg bg-surface/60 font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60">
                {#each [32,64,128,256] as v}<option value={v}>{v}</option>{/each}
              </select>
            </div>

            <div class="space-y-1">
              <label class="text-xs font-mono text-muted-foreground">learning_rate</label>
              <input type="number" bind:value={trainCfg.learning_rate} step="0.00001" min="0.00001" max="0.01"
                class="w-full px-2 py-1 text-xs border border-border/60 rounded-lg bg-surface/60 font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60"/>
            </div>

            <div class="space-y-1">
              <div class="flex justify-between text-xs font-mono">
                <label class="text-muted-foreground">warmup_steps</label>
                <span class="text-foreground font-medium">{trainCfg.warmup_steps}</span>
              </div>
              <input type="range" min="100" max="2000" step="100" bind:value={trainCfg.warmup_steps} class="w-full accent-cyan-400 h-1.5 rounded"/>
            </div>

            <div class="space-y-1">
              <div class="flex justify-between text-xs font-mono">
                <label class="text-muted-foreground">weight_decay</label>
                <span class="text-foreground font-medium">{trainCfg.weight_decay.toFixed(2)}</span>
              </div>
              <input type="range" min="0.01" max="0.1" step="0.01" bind:value={trainCfg.weight_decay} class="w-full accent-cyan-400 h-1.5 rounded"/>
            </div>

            <div class="space-y-1">
              <div class="flex justify-between text-xs font-mono">
                <label class="text-muted-foreground">grad_clip</label>
                <span class="text-foreground font-medium">{trainCfg.grad_clip.toFixed(1)}</span>
              </div>
              <input type="range" min="0.5" max="2.0" step="0.1" bind:value={trainCfg.grad_clip} class="w-full accent-cyan-400 h-1.5 rounded"/>
            </div>

            <div class="space-y-1">
              <label class="text-xs font-mono text-muted-foreground">optimizer</label>
              <select bind:value={trainCfg.optimizer} class="w-full px-2 py-1 text-xs border border-border/60 rounded-lg bg-surface/60 font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60">
                <option value="adamw">AdamW</option>
                <option value="adam">Adam</option>
                <option value="sgd">SGD</option>
              </select>
            </div>

            <div class="space-y-1">
              <label class="text-xs font-mono text-muted-foreground">lr_schedule</label>
              <select bind:value={trainCfg.lr_schedule} class="w-full px-2 py-1 text-xs border border-border/60 rounded-lg bg-surface/60 font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60">
                <option value="cosine">Cosine Annealing</option>
                <option value="linear">Linear</option>
                <option value="constant">Constant</option>
              </select>
            </div>

            <div class="space-y-1">
              <div class="flex justify-between text-xs font-mono">
                <label for="train-val-split" class="text-muted-foreground">validation_split</label>
                <span class="text-foreground font-medium">{Math.round(trainCfg.validation_split * 100)}%</span>
              </div>
              <input id="train-val-split" type="range" min="0.1" max="0.3" step="0.05" bind:value={trainCfg.validation_split} class="w-full accent-cyan-400 h-1.5 rounded"/>
            </div>

            <!-- Augmentations -->
            <div class="pt-1 border-t border-border/40 space-y-2" role="group" aria-labelledby="train-aug-label">
              <span id="train-aug-label" class="text-xs font-mono text-muted-foreground uppercase tracking-wider">Augmentation</span>
              <label class="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" bind:checked={trainCfg.aug_random_crop} class="accent-cyan-400 w-3.5 h-3.5"/>
                <span class="text-xs font-mono text-foreground">Random Crop</span>
              </label>
              <label class="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" bind:checked={trainCfg.aug_horizontal_flip} class="accent-cyan-400 w-3.5 h-3.5"/>
                <span class="text-xs font-mono text-foreground">Horizontal Flip</span>
              </label>
              <label class="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" bind:checked={trainCfg.aug_color_jitter} class="accent-cyan-400 w-3.5 h-3.5"/>
                <span class="text-xs font-mono text-foreground">Color Jitter</span>
              </label>
              <label class="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" bind:checked={trainCfg.aug_mixup} class="accent-cyan-400 w-3.5 h-3.5"/>
                <span class="text-xs font-mono text-foreground">Mixup</span>
              </label>
            </div>

            <!-- 3D Graph update frequency -->
            <div class="pt-1 border-t border-border/40 space-y-1" role="group" aria-labelledby="train-graph-label">
              <span id="train-graph-label" class="text-xs font-mono text-muted-foreground uppercase tracking-wider">3D Graph</span>
              <div class="flex items-center gap-2">
                <label for="train-graph-batches" class="text-[10px] font-mono text-muted-foreground whitespace-nowrap">update every</label>
                <input id="train-graph-batches" type="number" min="0" max="200" step="1"
                  bind:value={trainCfg.graph_update_every_n_batches}
                  class="w-16 text-xs font-mono bg-surface/60 border border-border/60 rounded-lg px-2 py-0.5 text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60"/>
                <span class="text-[10px] font-mono text-muted-foreground/60">batches (0 = epoch-end only)</span>
              </div>
            </div>
          </div>

        <!-- ── LIVE STATS TAB ── -->
        {:else}
          <h3 class="font-serif text-sm text-foreground">Live Training Stats</h3>

          <!-- Status + ETA -->
          <div class="bg-surface/40 backdrop-blur-md border border-border/40 rounded-xl p-3 space-y-2">
            <div class="flex items-center justify-between text-xs font-mono">
              <div class="flex items-center gap-1.5 flex-wrap">
                <div class="w-2 h-2 rounded-full flex-shrink-0
                  {jobStatus === 'idle' ? 'bg-muted-foreground/40' :
                   (jobStatus === 'training' || jobStatus === 'merging' || jobStatus === 'finetuning') ? 'bg-amber-400 animate-pulse' :
                   jobStatus === 'done' ? 'bg-emerald-400' : 'bg-destructive'}"
                />
                <span class="text-foreground capitalize font-medium">{jobStatus}</span>
                {#if jobStatus === 'training' && currentTrainingSpecName}
                  <span class="text-muted-foreground/80">· {currentTrainingSpecName}</span>
                {:else if jobStatus === 'merging'}
                  <span class="text-muted-foreground/80">· merging</span>
                {:else if jobStatus === 'finetuning'}
                  <span class="text-muted-foreground/80">· fine-tune</span>
                {/if}
              </div>
              {#if etaSeconds > 0 && (jobStatus === 'training' || jobStatus === 'finetuning')}
                <span class="text-muted-foreground">ETA {formatETA(etaSeconds)}</span>
              {/if}
            </div>

            {#if totalEpochs > 0}
              <div>
                <div class="flex justify-between text-[10px] font-mono text-muted-foreground mb-1">
                  <span>
                    {#if jobStatus === 'training' && currentTrainingSpecName}
                      {currentTrainingSpecName}: {currentEpoch}/{totalEpochs}
                    {:else if jobStatus === 'merging'}
                      merging…
                    {:else if jobStatus === 'finetuning'}
                      fine-tune: {currentEpoch}/{totalEpochs}
                    {:else}
                      epoch {currentEpoch}/{totalEpochs}
                    {/if}
                  </span>
                  <span>{Math.round(currentEpoch/totalEpochs * 100)}%</span>
                </div>
                <div class="w-full bg-border/40 rounded-full h-1.5">
                  <div class="bg-amber-400 h-1.5 rounded-full transition-all duration-500"
                    style="width:{totalEpochs > 0 ? currentEpoch/totalEpochs*100 : 0}%"/>
                </div>
              </div>
            {/if}
          </div>

          <!-- Live chart -->
          {#if trainLosses.length > 1 || valAccs.length > 1}
            <div class="border border-border/40 rounded-xl p-2 bg-surface/40 backdrop-blur-md">
              <ProgressChart losses={trainLosses} valAccs={valAccs} width={272} height={110}/>
            </div>
          {/if}

          <!-- Log terminal -->
          {#if logLines.length > 0}
            <div role="group" aria-labelledby="live-log-label">
              <span id="live-log-label" class="text-xs font-mono text-muted-foreground uppercase tracking-wider">Log</span>
              <div class="mt-1.5 bg-surface/50 backdrop-blur-sm border border-border/40 rounded-lg p-2.5 h-52 overflow-y-auto font-mono text-[11px] text-green-400 leading-relaxed scroll-smooth scrollbar-thin" id="log-box">
                {#each logLines as line}
                  <div class:text-red-400={line.startsWith('✗')} class:text-emerald-400={line.startsWith('✓')} class:text-amber-300={line.startsWith('━')}>
                    {line}
                  </div>
                {/each}
              </div>
            </div>
          {:else}
            <div class="text-center py-8 text-muted-foreground font-mono text-xs">
              <div class="text-2xl mb-2">⬡</div>
              Click <span class="text-amber-400 font-medium">▶ Run Pipeline</span> to start
            </div>
          {/if}

        {/if}
      </div>

      <!-- Persistent mini chart (always visible during/after training) -->
      {#if trainLosses.length > 1 || valAccs.length > 1}
        <div class="border-t border-border/40 px-4 py-3 bg-surface/40 backdrop-blur-md flex-shrink-0" role="img" aria-label="Training curve: loss and validation accuracy">
          <div class="flex justify-between text-[10px] font-mono text-muted-foreground mb-1.5">
            <span>Training curve</span>
            <span><span class="text-amber-400">—</span> loss &nbsp; <span class="text-emerald-400">- -</span> val%</span>
          </div>
          <ProgressChart losses={trainLosses} valAccs={valAccs} width={272} height={80}/>
        </div>
      {/if}
    </aside>
      </div><!-- /studio-panel-content right -->
    </div><!-- /studio-panel-wrapper right -->
    {#if isRightMinimized}
      <button class="studio-expand-btn studio-expand-right" on:click={() => isRightMinimized = false} aria-label="Expand right panel">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
      </button>
    {/if}

    <!-- ════ BOTTOM FLOATING: 3D BDH Weight Graph ════════════════ -->
    {#if showGraph || graphSnapshot}
      <div class="studio-panel-wrapper studio-panel-bottom" class:minimized={isGraphMinimized}>
        <div class="studio-panel-header">
          <div class="flex items-center gap-2">
            <span class="text-primary">⬡</span>
            <span class="studio-panel-title" style="text-transform:none; letter-spacing:0; font-size:0.7rem">3D BDH Weight Graph</span>
            <span class="text-[10px] font-mono text-muted-foreground/60">— most-active neurons · edges = encoder weights</span>
          </div>
          <div class="flex items-center gap-1">
            <button class="studio-panel-toggle" on:click={() => isGraphMinimized = !isGraphMinimized} aria-label="Toggle graph">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {#if isGraphMinimized}
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7"/>
                {:else}
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                {/if}
              </svg>
            </button>
            <button class="studio-panel-close" on:click={() => { showGraph = false; isGraphMinimized = false; }} aria-label="Close graph">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
            </button>
          </div>
        </div>
        <div class="studio-panel-content studio-graph-panel-content" style="height:300px; overflow:hidden; flex-shrink:0;">
          <TrainingGraph3D
            snapshot={graphSnapshot}
            epoch={graphEpoch}
            batch={graphBatch}
            loss={graphLoss}
            valAcc={graphValAcc}
            nLayers={archCfg.n_layer}
            {inferMode}
          />
        </div>
      </div>
    {/if}

    <!-- Bottom-edge expand tab — reopen graph after closing OR when minimized -->
    {#if graphEverShown && (!showGraph || isGraphMinimized)}
      <button
        class="studio-expand-btn studio-expand-bottom"
        on:click={() => { showGraph = true; isGraphMinimized = false; }}
        aria-label="Reopen 3D graph"
        title="Reopen 3D weight graph"
      >
        <svg class="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7"/>
        </svg>
        <span>3D Graph</span>
      </button>
    {/if}

  </div><!-- /relative canvas container -->

  <!-- ════ INFERENCE PANEL ══════════════════════════════════════════ -->
  {#if availableCheckpoints.length > 0 || Object.keys(JOBS_CACHE).length > 0}
    <div class="border-t border-border/30 bg-background flex-shrink-0">
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-1.5 border-b border-border/30">
        <div class="flex items-center gap-2">
          <span class="text-emerald-400 text-sm">◎</span>
          <span class="font-mono text-xs text-foreground font-medium text-glow-cyan">Inference</span>
          <span class="text-[10px] font-mono text-muted-foreground/50">— run image through trained model</span>
        </div>
        {#if inferMode}
          <button on:click={exitInferMode}
            class="font-mono text-xs text-amber-400 hover:text-amber-300 transition-colors">
            ← back to training view
          </button>
        {/if}
      </div>

      <div class="flex gap-4 px-4 py-3 items-start flex-wrap">
        <!-- Checkpoint selector -->
        <div class="space-y-1 min-w-[200px]">
          <label class="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">checkpoint</label>
          <select bind:value={inferCheckpoint}
            class="w-full text-xs font-mono bg-surface/60 border border-border/60 rounded-lg px-2 py-1.5 text-foreground focus:border-primary/60 focus:outline-none focus:ring-1 focus:ring-primary/40">
            <option value={null}>— select model —</option>
            {#each availableCheckpoints as ckpt}
              <option value={ckpt.path}>{ckpt.label}</option>
            {/each}
          </select>
        </div>

        <!-- Image upload -->
        <div class="space-y-1">
          <label class="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">image (32×32 CIFAR)</label>
          <div class="flex items-center gap-2">
            {#if inferImagePreview}
              <img src={inferImagePreview} alt="input" class="w-10 h-10 rounded border border-border/50 object-cover pixelated"/>
            {:else}
              <div class="w-10 h-10 rounded border border-dashed border-border/50 flex items-center justify-center text-muted-foreground/40 text-xs">img</div>
            {/if}
            <label class="cursor-pointer font-mono text-xs px-2.5 py-1 bg-surface/60 hover:bg-surface-hover border border-border/50 rounded-lg text-foreground transition-colors">
              upload
              <input type="file" accept="image/*" class="hidden" on:change={handleInferImageUpload}/>
            </label>
          </div>
        </div>

        <!-- Run button -->
        <div class="flex items-end">
          <button on:click={runInference}
            disabled={!inferCheckpoint || !inferImageB64 || inferLoading}
            class="font-mono text-xs px-4 py-1.5 rounded-lg transition-colors
              {inferLoading ? 'bg-surface/60 border border-border/40 text-muted-foreground/50 cursor-wait'
              : (!inferCheckpoint || !inferImageB64) ? 'bg-surface/40 border border-border/30 text-muted-foreground/40 cursor-not-allowed'
              : 'bg-emerald-400/20 hover:bg-emerald-400/30 border border-emerald-400/60 text-emerald-400 cursor-pointer'}">
            {inferLoading ? '⏳ running…' : '▶ Run'}
          </button>
        </div>

        <!-- Results: confidence bars -->
        {#if inferResult}
          <div class="flex-1 min-w-[260px] space-y-1">
            <div class="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-1.5">
              prediction:
              <span class="text-emerald-400 font-bold ml-1">
                {inferResult.class_names[inferResult.predicted_class] ?? `class ${inferResult.predicted_class}`}
              </span>
            </div>
            {#each inferResult.class_probs as prob, i}
              {@const pct = Math.round(prob * 100)}
              {@const isTop = i === inferResult.predicted_class}
              <div class="flex items-center gap-2 text-[10px] font-mono">
                <span class="w-20 text-right {isTop ? 'text-emerald-400 font-bold' : 'text-muted-foreground/50'} truncate">
                  {inferResult.class_names[i] ?? `cls ${i}`}
                </span>
                <div class="flex-1 bg-border/30 rounded-sm h-2 overflow-hidden">
                  <div class="h-full rounded-sm transition-all duration-500
                    {isTop ? 'bg-emerald-400' : 'bg-muted-foreground/30'}"
                    style="width: {pct}%"/>
                </div>
                <span class="w-8 text-right {isTop ? 'text-emerald-400' : 'text-muted-foreground/40'}">{pct}%</span>
              </div>
            {/each}
          </div>
        {/if}

        {#if inferError}
          <div class="text-xs font-mono text-destructive max-w-sm">{inferError}</div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .canvas-grid {
    background-image: radial-gradient(circle, hsl(240 20% 28% / 0.5) 1px, transparent 1px);
    background-size: 24px 24px;
  }

  input[type="range"] {
    cursor: pointer;
  }

  #log-box > div { transition: none; }

  /* ─── Studio floating panels ─────────────────────────────────── */
  .studio-panel-wrapper {
    position: absolute;
    z-index: 20;
    background: hsl(var(--surface) / 0.4);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid hsl(var(--border) / 0.5);
    border-radius: 0.75rem;
    overflow: hidden;
    transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
  }

  /* Left panel */
  .studio-panel-left {
    top: 8px;
    left: 8px;
    width: 288px;
    max-height: calc(100% - 16px);
  }
  .studio-panel-left.minimized {
    transform: translateX(calc(-100% - 12px));
  }

  /* Right panel */
  .studio-panel-right {
    top: 8px;
    right: 8px;
    width: 320px;
    max-height: calc(100% - 16px);
  }
  .studio-panel-right.minimized {
    transform: translateX(calc(100% + 12px));
  }

  /* Bottom graph panel */
  .studio-panel-bottom {
    bottom: 8px;
    left: 50%;
    transform: translateX(-50%);
    width: 90%;
    max-width: 960px;
    height: 344px; /* fixed: header 44px + content 300px */
  }
  .studio-panel-bottom.minimized {
    transform: translate(-50%, calc(100% + 12px));
  }

  /* Panel header */
  .studio-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0.75rem 0.4rem 1rem;
    border-bottom: 1px solid hsl(var(--border) / 0.3);
    background: rgba(0, 0, 0, 0.15);
    flex-shrink: 0;
  }
  .studio-panel-title {
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: hsl(var(--muted-foreground));
    font-family: 'JetBrains Mono', monospace;
  }
  .studio-panel-toggle,
  .studio-panel-close {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 1.5rem;
    height: 1.5rem;
    border: none;
    border-radius: 0.375rem;
    background: transparent;
    color: hsl(var(--muted-foreground));
    cursor: pointer;
    transition: color 0.2s, background 0.2s;
  }
  .studio-panel-toggle:hover,
  .studio-panel-close:hover {
    color: hsl(var(--foreground));
    background: hsl(var(--border) / 0.3);
  }
  .studio-panel-content {
    overflow-y: auto;
    flex: 1;
    scrollbar-width: thin;
    scrollbar-color: hsl(215 20% 35%) transparent;
  }
  .studio-panel-content::-webkit-scrollbar { width: 5px; }
  .studio-panel-content::-webkit-scrollbar-track { background: transparent; }
  .studio-panel-content::-webkit-scrollbar-thumb {
    background-color: hsl(215 20% 35%);
    border-radius: 3px;
  }

  /* Graph panel content: keep transparent so glass layer inside TrainingGraph3D shows */
  .studio-graph-panel-content {
    background: transparent;
  }

  /* Edge expand tabs */
  .studio-expand-btn {
    position: absolute;
    z-index: 21;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 1.25rem;
    height: 2.5rem;
    border: 1px solid hsl(var(--border) / 0.5);
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: hsl(var(--muted-foreground));
    cursor: pointer;
    transition: color 0.2s, background 0.2s;
  }
  .studio-expand-btn:hover {
    color: hsl(var(--foreground));
    background: rgba(0, 0, 0, 0.75);
  }
  .studio-expand-left {
    top: 50%;
    left: 0;
    transform: translateY(-50%);
    border-radius: 0 0.375rem 0.375rem 0;
    border-left: none;
  }
  .studio-expand-right {
    top: 50%;
    right: 0;
    transform: translateY(-50%);
    border-radius: 0.375rem 0 0 0.375rem;
    border-right: none;
  }
  .studio-expand-bottom {
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    border-radius: 0.375rem 0.375rem 0 0;
    border-bottom: none;
    width: auto;
    height: 1.75rem;
    padding: 0 0.75rem;
    gap: 0.35rem;
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
    white-space: nowrap;
  }

</style>
