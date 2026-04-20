const state = {
  images: [],
  imageName: null,
  image: null,
  rois: [],
  points: [],
  lines: [],
  selected: { type: null, index: null },
  tool: "roi",
  pointKind: "tube_top",
  lineKind: "horizontal_ref",
  fitScale: 1,
  zoom: 1,
  minZoom: 0.35,
  maxZoom: 16,
  offsetX: 0,
  offsetY: 0,
  draft: null,
  dragStart: null,
  dragMode: null,
  dragTarget: null,
  panStartOffset: null,
  pendingLineStart: null,
  pendingLineHover: null,
  homographyQuad: null,
  srcPointsOverride: null,
  dstRectOverride: null,
  overlayPreviewDrag: null,
  previewRect: null,
  previewHomographyMatrix: null,
  previewInverseHomographyMatrix: null,
  previewBaseSize: null,
  previewMode: "homography",
};

const POINT_KIND_META = {
  tube_top: { color: "#65d9ff", short: "top" },
  tube_base: { color: "#ffd95a", short: "base" },
  mark: { color: "#ff8bc4", short: "mark" },
};

const LINE_KIND_META = {
  horizontal_ref: { color: "#89d1ff", short: "horiz" },
  vertical_ref: { color: "#ff7878", short: "vref" },
  tube_axis: { color: "#ffb48f", short: "axis" },
  custom: { color: "#98ffab", short: "line" },
};

const imageSelect = document.getElementById("imageSelect");
const saveBtn = document.getElementById("saveBtn");
const reloadBtn = document.getElementById("reloadBtn");
const deleteBtn = document.getElementById("deleteBtn");
const clearBtn = document.getElementById("clearBtn");
const homographyBtn = document.getElementById("homographyBtn");
const tubeDetectionBtn = document.getElementById("tubeDetectionBtn");
const propagateBtn = document.getElementById("propagateBtn");
const downloadBtn = document.getElementById("downloadBtn");
const roiList = document.getElementById("roiList");
const pointList = document.getElementById("pointList");
const lineList = document.getElementById("lineList");
const statusBox = document.getElementById("status");
const selectedMeta = document.getElementById("selectedMeta");
const roiCanvas = document.getElementById("roiCanvas");
const roiCtx = roiCanvas.getContext("2d");
const cropCanvas = document.getElementById("cropCanvas");
const cropCtx = cropCanvas.getContext("2d");
const currentImageName = document.getElementById("currentImageName");
const imageMeta = document.getElementById("imageMeta");
const annotationHero = document.getElementById("annotationHero");
const roiCountBadge = document.getElementById("roiCountBadge");
const pointCountBadge = document.getElementById("pointCountBadge");
const lineCountBadge = document.getElementById("lineCountBadge");
const canvasMeta = document.getElementById("canvasMeta");
const canvasWrap = document.querySelector(".canvas-wrap");
const layoutRoot = document.querySelector(".layout");
const toolRoiBtn = document.getElementById("toolRoiBtn");
const toolPointBtn = document.getElementById("toolPointBtn");
const toolLineBtn = document.getElementById("toolLineBtn");
const pointKindSelect = document.getElementById("pointKindSelect");
const pointLabelInput = document.getElementById("pointLabelInput");
const lineKindSelect = document.getElementById("lineKindSelect");
const lineLabelInput = document.getElementById("lineLabelInput");
const lineRealDistanceInput = document.getElementById("lineRealDistanceInput");
const selectedLabelInput = document.getElementById("selectedLabelInput");
const selectedLineRealDistanceInput = document.getElementById("selectedLineRealDistanceInput");
const annotationNoteInput = document.getElementById("annotationNoteInput");
const annotationHint = document.getElementById("annotationHint");
const clearPreviewBtn = document.getElementById("clearPreviewBtn");
const previewMeta = document.getElementById("previewMeta");
const toggleRightSidebarBtn = document.getElementById("toggleRightSidebarBtn");
const previewOverlayViewport = document.getElementById("previewOverlayViewport");
const previewOverlayCanvas = document.getElementById("previewOverlayCanvas");
const previewOverlayCtx = previewOverlayCanvas.getContext("2d");
const previewWarpViewport = document.getElementById("previewWarpViewport");
const previewOverlayImg = document.getElementById("previewOverlayImg");
const previewWarpImg = document.getElementById("previewWarpImg");
const previewViewers = {
  overlay: createPreviewViewer(previewOverlayViewport, previewOverlayImg),
  warp: createPreviewViewer(previewWarpViewport, previewWarpImg),
};

function setStatus(message) {
  statusBox.textContent = message;
}

function setRightSidebarCollapsed(collapsed) {
  if (!layoutRoot || !toggleRightSidebarBtn) {
    return;
  }
  layoutRoot.classList.toggle("right-collapsed", collapsed);
  toggleRightSidebarBtn.textContent = collapsed ? "Mostrar editor" : "Ocultar editor";
  window.requestAnimationFrame(() => {
    if (state.image) {
      resizeCanvasToViewport();
      drawCanvas();
    }
    Object.values(previewViewers).forEach((viewer) => {
      if (hasPreviewImage(viewer)) {
        resetPreviewViewer(viewer);
      }
    });
    drawHomographyQuadOverlay();
  });
}

function isTypingTarget(target) {
  if (!target) {
    return false;
  }

  const tagName = (target.tagName || "").toUpperCase();
  return (
    tagName === "INPUT" ||
    tagName === "TEXTAREA" ||
    tagName === "SELECT" ||
    target.isContentEditable === true
  );
}

function getZoomText() {
  return `${Math.round(state.zoom * 100)}%`;
}

function createPreviewViewer(viewport, img) {
  return {
    viewport,
    img,
    zoom: 1,
    minZoom: 1,
    maxZoom: 24,
    baseWidth: 0,
    baseHeight: 0,
    offsetX: 0,
    offsetY: 0,
    dragStart: null,
    startOffset: null,
    pointerId: null,
  };
}

function hasPreviewImage(viewer) {
  return Boolean(viewer?.img?.getAttribute("src"));
}

function applyPreviewTransform(viewer) {
  if (!viewer || !viewer.viewport || !viewer.img || viewer.baseWidth <= 0 || viewer.baseHeight <= 0) {
    return;
  }

  viewer.img.style.width = `${viewer.baseWidth}px`;
  viewer.img.style.height = `${viewer.baseHeight}px`;
  viewer.img.style.transform = `translate(${viewer.offsetX}px, ${viewer.offsetY}px) scale(${viewer.zoom})`;
  viewer.viewport.classList.toggle("has-image", hasPreviewImage(viewer));
  if (viewer === previewViewers.overlay) {
    drawHomographyQuadOverlay();
  }
}

function resetPreviewViewer(viewer) {
  if (!viewer || !viewer.viewport || !viewer.img || !viewer.img.naturalWidth || !viewer.img.naturalHeight) {
    return;
  }

  const viewportWidth = Math.max(120, viewer.viewport.clientWidth);
  const viewportHeight = Math.max(120, viewer.viewport.clientHeight);
  const fit = Math.min(viewportWidth / viewer.img.naturalWidth, viewportHeight / viewer.img.naturalHeight);

  viewer.baseWidth = Math.max(1, viewer.img.naturalWidth * fit);
  viewer.baseHeight = Math.max(1, viewer.img.naturalHeight * fit);
  viewer.zoom = 1;
  viewer.offsetX = (viewportWidth - viewer.baseWidth) / 2;
  viewer.offsetY = (viewportHeight - viewer.baseHeight) / 2;
  applyPreviewTransform(viewer);
}

function clearPreviewViewer(viewer) {
  if (!viewer || !viewer.viewport || !viewer.img) {
    return;
  }

  viewer.img.removeAttribute("src");
  viewer.img.style.width = "0px";
  viewer.img.style.height = "0px";
  viewer.img.style.transform = "";
  viewer.zoom = 1;
  viewer.baseWidth = 0;
  viewer.baseHeight = 0;
  viewer.offsetX = 0;
  viewer.offsetY = 0;
  viewer.dragStart = null;
  viewer.startOffset = null;
  viewer.pointerId = null;
  viewer.viewport.classList.remove("has-image", "dragging");
  if (viewer === previewViewers.overlay) {
    resizePreviewOverlayCanvas();
    previewOverlayCtx.clearRect(0, 0, previewOverlayCanvas.width, previewOverlayCanvas.height);
  }
}

function zoomPreviewViewer(viewer, clientX, clientY, factor) {
  if (!viewer || !viewer.viewport || !viewer.img || viewer.baseWidth <= 0 || viewer.baseHeight <= 0) {
    return;
  }

  const rect = viewer.viewport.getBoundingClientRect();
  const px = clientX - rect.left;
  const py = clientY - rect.top;
  const oldZoom = viewer.zoom;
  const nextZoom = Math.max(viewer.minZoom, Math.min(viewer.maxZoom, viewer.zoom * factor));
  if (nextZoom === oldZoom) {
    return;
  }

  viewer.offsetX = px - ((px - viewer.offsetX) / oldZoom) * nextZoom;
  viewer.offsetY = py - ((py - viewer.offsetY) / oldZoom) * nextZoom;
  viewer.zoom = nextZoom;
  applyPreviewTransform(viewer);
}

function bindPreviewViewer(viewer) {
  if (!viewer?.viewport || !viewer?.img) {
    return;
  }

  viewer.img.addEventListener("load", () => resetPreviewViewer(viewer));

  viewer.viewport.addEventListener(
    "wheel",
    (event) => {
      if (!hasPreviewImage(viewer)) {
        return;
      }
      event.preventDefault();
      zoomPreviewViewer(viewer, event.clientX, event.clientY, event.deltaY < 0 ? 1.12 : 1 / 1.12);
    },
    { passive: false },
  );

  viewer.viewport.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !hasPreviewImage(viewer)) {
      return;
    }
    viewer.pointerId = event.pointerId;
    viewer.dragStart = [event.clientX, event.clientY];
    viewer.startOffset = [viewer.offsetX, viewer.offsetY];
    viewer.viewport.setPointerCapture(event.pointerId);
    viewer.viewport.classList.add("dragging");
  });

  viewer.viewport.addEventListener("pointermove", (event) => {
    if (viewer.pointerId !== event.pointerId || !viewer.dragStart || !viewer.startOffset) {
      return;
    }
    viewer.offsetX = viewer.startOffset[0] + (event.clientX - viewer.dragStart[0]);
    viewer.offsetY = viewer.startOffset[1] + (event.clientY - viewer.dragStart[1]);
    applyPreviewTransform(viewer);
  });

  const stopDragging = (event) => {
    if (viewer.pointerId !== event.pointerId) {
      return;
    }
    viewer.pointerId = null;
    viewer.dragStart = null;
    viewer.startOffset = null;
    viewer.viewport.classList.remove("dragging");
    if (viewer.viewport.hasPointerCapture(event.pointerId)) {
      viewer.viewport.releasePointerCapture(event.pointerId);
    }
  };

  viewer.viewport.addEventListener("pointerup", stopDragging);
  viewer.viewport.addEventListener("pointercancel", stopDragging);
  viewer.viewport.addEventListener("dblclick", () => {
    if (hasPreviewImage(viewer)) {
      resetPreviewViewer(viewer);
    }
  });
}

function getOverlayImageScale() {
  const viewer = previewViewers.overlay;
  if (!viewer.baseWidth || !previewOverlayImg.naturalWidth) {
    return 1;
  }
  return viewer.baseWidth / previewOverlayImg.naturalWidth;
}

function imagePointToOverlayCanvas(point) {
  const viewer = previewViewers.overlay;
  const scale = getOverlayImageScale();
  return {
    x: viewer.offsetX + point.x * scale * viewer.zoom,
    y: viewer.offsetY + point.y * scale * viewer.zoom,
  };
}

function overlayCanvasToImagePoint(canvasX, canvasY) {
  const viewer = previewViewers.overlay;
  const scale = getOverlayImageScale();
  return {
    x: (canvasX - viewer.offsetX) / (scale * viewer.zoom),
    y: (canvasY - viewer.offsetY) / (scale * viewer.zoom),
  };
}

function applyHomography(matrix, point) {
  if (!Array.isArray(matrix) || matrix.length !== 3) {
    return null;
  }
  const x = Number(point.x);
  const y = Number(point.y);
  const w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
  if (!Number.isFinite(w) || Math.abs(w) < 1e-9) {
    return null;
  }
  return {
    x: (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / w,
    y: (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / w,
  };
}

function normalizePreviewRect(rect) {
  if (!rect) {
    return null;
  }
  const minSize = 40;
  let [x0, y0, x1, y1] = rect.map(Number);
  if (x1 < x0 + minSize) {
    x1 = x0 + minSize;
  }
  if (y1 < y0 + minSize) {
    y1 = y0 + minSize;
  }
  return [x0, y0, x1, y1];
}

function computeProjectedPreviewQuad() {
  if (!state.previewRect || !state.previewInverseHomographyMatrix) {
    return null;
  }
  const [x0, y0, x1, y1] = state.previewRect;
  const rectPoints = [
    { x: x0, y: y0 },
    { x: x1, y: y0 },
    { x: x0, y: y1 },
    { x: x1, y: y1 },
  ];
  return rectPoints
    .map((point) => applyHomography(state.previewInverseHomographyMatrix, point))
    .filter(Boolean);
}

function getHomographyQuadCanvasPoints() {
  if (!state.homographyQuad || state.homographyQuad.length !== 4) {
    return null;
  }
  const points = state.homographyQuad.map(imagePointToOverlayCanvas);
  return {
    tl: points[0],
    tr: points[1],
    bl: points[2],
    br: points[3],
    ordered: [points[0], points[1], points[3], points[2]],
  };
}

function distancePointToSegment(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq <= 1e-9) {
    return Math.hypot(point.x - start.x, point.y - start.y);
  }
  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSq));
  const projX = start.x + t * dx;
  const projY = start.y + t * dy;
  return Math.hypot(point.x - projX, point.y - projY);
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-9) + xi;
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

function resizePreviewOverlayCanvas() {
  const rect = previewOverlayViewport.getBoundingClientRect();
  previewOverlayCanvas.width = Math.max(1, Math.round(rect.width));
  previewOverlayCanvas.height = Math.max(1, Math.round(rect.height));
}

function drawHomographyQuadOverlay() {
  resizePreviewOverlayCanvas();
  previewOverlayCtx.clearRect(0, 0, previewOverlayCanvas.width, previewOverlayCanvas.height);

  if (!hasPreviewImage(previewViewers.overlay) || !state.homographyQuad || state.homographyQuad.length !== 4) {
    return;
  }

  const quad = getHomographyQuadCanvasPoints();
  if (!quad) {
    return;
  }
  const points = [quad.tl, quad.tr, quad.bl, quad.br];
  const drawOrder = quad.ordered;
  previewOverlayCtx.save();
  previewOverlayCtx.lineWidth = 3;
  previewOverlayCtx.strokeStyle = "#ffe26a";
  previewOverlayCtx.fillStyle = "rgba(255, 226, 106, 0.14)";
  previewOverlayCtx.beginPath();
  previewOverlayCtx.moveTo(drawOrder[0].x, drawOrder[0].y);
  drawOrder.slice(1).forEach((point) => previewOverlayCtx.lineTo(point.x, point.y));
  previewOverlayCtx.closePath();
  previewOverlayCtx.fill();
  previewOverlayCtx.stroke();

  previewOverlayCtx.strokeStyle = "rgba(255, 226, 106, 0.9)";
  previewOverlayCtx.lineWidth = 8;
  previewOverlayCtx.beginPath();
  previewOverlayCtx.moveTo(quad.tl.x, quad.tl.y);
  previewOverlayCtx.lineTo(quad.tr.x, quad.tr.y);
  previewOverlayCtx.moveTo(quad.tr.x, quad.tr.y);
  previewOverlayCtx.lineTo(quad.br.x, quad.br.y);
  previewOverlayCtx.moveTo(quad.br.x, quad.br.y);
  previewOverlayCtx.lineTo(quad.bl.x, quad.bl.y);
  previewOverlayCtx.moveTo(quad.bl.x, quad.bl.y);
  previewOverlayCtx.lineTo(quad.tl.x, quad.tl.y);
  previewOverlayCtx.stroke();

  points.forEach((point, index) => {
    previewOverlayCtx.fillStyle = "#65d9ff";
    previewOverlayCtx.beginPath();
    previewOverlayCtx.arc(point.x, point.y, 7, 0, Math.PI * 2);
    previewOverlayCtx.fill();
    previewOverlayCtx.strokeStyle = "#081017";
    previewOverlayCtx.lineWidth = 2;
    previewOverlayCtx.stroke();
    previewOverlayCtx.fillStyle = "#eef4f8";
    previewOverlayCtx.font = '700 12px "Space Grotesk", sans-serif';
    previewOverlayCtx.fillText(["TL", "TR", "BL", "BR"][index], point.x + 10, point.y - 10);
  });
  previewOverlayCtx.restore();
}

function hitTestHomographyHandle(clientX, clientY) {
  if (!state.homographyQuad || state.homographyQuad.length !== 4) {
    return null;
  }

  const rect = previewOverlayCanvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  let best = null;

  const quad = getHomographyQuadCanvasPoints();
  if (!quad) {
    return null;
  }

  [quad.tl, quad.tr, quad.bl, quad.br].forEach((canvasPoint, index) => {
    const distance = Math.hypot(canvasPoint.x - x, canvasPoint.y - y);
    if (distance <= 16 && (!best || distance < best.distance)) {
      best = { index, distance };
    }
  });

  return best;
}

function hitTestHomographyEdge(clientX, clientY) {
  const quad = getHomographyQuadCanvasPoints();
  if (!quad) {
    return null;
  }

  const rect = previewOverlayCanvas.getBoundingClientRect();
  const point = { x: clientX - rect.left, y: clientY - rect.top };
  const edges = [
    { name: "top", start: quad.tl, end: quad.tr },
    { name: "right", start: quad.tr, end: quad.br },
    { name: "bottom", start: quad.bl, end: quad.br },
    { name: "left", start: quad.tl, end: quad.bl },
  ];

  let best = null;
  edges.forEach((edge) => {
    const distance = distancePointToSegment(point, edge.start, edge.end);
    if (distance <= 14 && (!best || distance < best.distance)) {
      best = { edge: edge.name, distance };
    }
  });
  return best;
}

function isInsideHomographyQuad(clientX, clientY) {
  const quad = getHomographyQuadCanvasPoints();
  if (!quad) {
    return false;
  }
  const rect = previewOverlayCanvas.getBoundingClientRect();
  const point = { x: clientX - rect.left, y: clientY - rect.top };
  return pointInPolygon(point, quad.ordered);
}

function updatePreviewRectHandle(index, imagePoint) {
  if (!state.previewRect || !state.previewHomographyMatrix) {
    return false;
  }

  const dstPoint = applyHomography(state.previewHomographyMatrix, imagePoint);
  if (!dstPoint) {
    return false;
  }

  const nextRect = [...state.previewRect];
  if (index === 0) {
    nextRect[0] = dstPoint.x;
    nextRect[1] = dstPoint.y;
  } else if (index === 1) {
    nextRect[2] = dstPoint.x;
    nextRect[1] = dstPoint.y;
  } else if (index === 2) {
    nextRect[0] = dstPoint.x;
    nextRect[3] = dstPoint.y;
  } else if (index === 3) {
    nextRect[2] = dstPoint.x;
    nextRect[3] = dstPoint.y;
  }

  state.previewRect = normalizePreviewRect(nextRect);
  state.homographyQuad = computeProjectedPreviewQuad();
  return Array.isArray(state.homographyQuad) && state.homographyQuad.length === 4;
}

function updatePreviewRectEdge(edge, imagePoint) {
  if (!state.previewRect || !state.previewHomographyMatrix) {
    return false;
  }
  const dstPoint = applyHomography(state.previewHomographyMatrix, imagePoint);
  if (!dstPoint) {
    return false;
  }
  const nextRect = [...state.previewRect];
  if (edge === "left") {
    nextRect[0] = dstPoint.x;
  } else if (edge === "right") {
    nextRect[2] = dstPoint.x;
  } else if (edge === "top") {
    nextRect[1] = dstPoint.y;
  } else if (edge === "bottom") {
    nextRect[3] = dstPoint.y;
  }
  state.previewRect = normalizePreviewRect(nextRect);
  state.homographyQuad = computeProjectedPreviewQuad();
  return Array.isArray(state.homographyQuad) && state.homographyQuad.length === 4;
}

function updatePreviewRectMove(imagePoint, startRect, startDstPoint) {
  if (!state.previewHomographyMatrix || !startRect || !startDstPoint) {
    return false;
  }
  const dstPoint = applyHomography(state.previewHomographyMatrix, imagePoint);
  if (!dstPoint) {
    return false;
  }
  const dx = dstPoint.x - startDstPoint.x;
  const dy = dstPoint.y - startDstPoint.y;
  state.previewRect = normalizePreviewRect([
    startRect[0] + dx,
    startRect[1] + dy,
    startRect[2] + dx,
    startRect[3] + dy,
  ]);
  state.homographyQuad = computeProjectedPreviewQuad();
  return Array.isArray(state.homographyQuad) && state.homographyQuad.length === 4;
}

function bindOverlayQuadEditor() {
  previewOverlayCanvas.addEventListener(
    "wheel",
    (event) => {
      if (!hasPreviewImage(previewViewers.overlay)) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      zoomPreviewViewer(previewViewers.overlay, event.clientX, event.clientY, event.deltaY < 0 ? 1.12 : 1 / 1.12);
    },
    { passive: false },
  );

  previewOverlayCanvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !hasPreviewImage(previewViewers.overlay)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();

    const handleHit = hitTestHomographyHandle(event.clientX, event.clientY);
    if (handleHit) {
      state.overlayPreviewDrag = {
        mode: "handle",
        index: handleHit.index,
        moved: false,
        pointerId: event.pointerId,
      };
      previewOverlayCanvas.setPointerCapture(event.pointerId);
      return;
    }

    const edgeHit = hitTestHomographyEdge(event.clientX, event.clientY);
    if (edgeHit) {
      state.overlayPreviewDrag = {
        mode: "edge",
        edge: edgeHit.edge,
        moved: false,
        pointerId: event.pointerId,
      };
      previewOverlayCanvas.setPointerCapture(event.pointerId);
      return;
    }

    if (isInsideHomographyQuad(event.clientX, event.clientY)) {
      const rect = previewOverlayCanvas.getBoundingClientRect();
      const imagePoint = overlayCanvasToImagePoint(event.clientX - rect.left, event.clientY - rect.top);
      const startDstPoint = applyHomography(state.previewHomographyMatrix, imagePoint);
      state.overlayPreviewDrag = {
        mode: "move",
        moved: false,
        pointerId: event.pointerId,
        startRect: state.previewRect ? [...state.previewRect] : null,
        startDstPoint,
      };
      previewOverlayCanvas.setPointerCapture(event.pointerId);
      return;
    }

    state.overlayPreviewDrag = {
      mode: "pan",
      pointerId: event.pointerId,
      start: [event.clientX, event.clientY],
      startOffset: [previewViewers.overlay.offsetX, previewViewers.overlay.offsetY],
    };
    previewOverlayCanvas.setPointerCapture(event.pointerId);
  });

  previewOverlayCanvas.addEventListener("pointermove", (event) => {
    const drag = state.overlayPreviewDrag;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();

    if (drag.mode === "pan") {
      previewViewers.overlay.offsetX = drag.startOffset[0] + (event.clientX - drag.start[0]);
      previewViewers.overlay.offsetY = drag.startOffset[1] + (event.clientY - drag.start[1]);
      applyPreviewTransform(previewViewers.overlay);
      return;
    }

    const rect = previewOverlayCanvas.getBoundingClientRect();
    const nextPoint = overlayCanvasToImagePoint(event.clientX - rect.left, event.clientY - rect.top);
    let updated = false;
    if (drag.mode === "handle") {
      updated = updatePreviewRectHandle(drag.index, nextPoint);
    } else if (drag.mode === "edge") {
      updated = updatePreviewRectEdge(drag.edge, nextPoint);
    } else if (drag.mode === "move") {
      updated = updatePreviewRectMove(nextPoint, drag.startRect, drag.startDstPoint);
    }
    if (updated) {
      drag.moved = true;
      drawHomographyQuadOverlay();
    }
  });

  const stopDrag = (event) => {
    const drag = state.overlayPreviewDrag;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const moved = (drag.mode === "handle" || drag.mode === "edge" || drag.mode === "move") && drag.moved;
    state.overlayPreviewDrag = null;
    if (previewOverlayCanvas.hasPointerCapture(event.pointerId)) {
      previewOverlayCanvas.releasePointerCapture(event.pointerId);
    }
    if (moved) {
      const runner = state.previewMode === "tubes" ? previewTubeDetection : previewHomography;
      runner().catch((err) => {
        previewMeta.textContent = `Error: ${String(err.message || err)}`;
        setStatus(String(err));
      });
    }
  };

  previewOverlayCanvas.addEventListener("pointerup", stopDrag);
  previewOverlayCanvas.addEventListener("pointercancel", stopDrag);
  previewOverlayCanvas.addEventListener("dblclick", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (hasPreviewImage(previewViewers.overlay)) {
      resetPreviewViewer(previewViewers.overlay);
      drawHomographyQuadOverlay();
    }
  });
}

function roiIdFromIndex(index) {
  if (index == null || index < 0) {
    return null;
  }
  return `roi_${String(index + 1).padStart(2, "0")}`;
}

function roiIndexFromId(roiId) {
  if (!roiId || typeof roiId !== "string") {
    return null;
  }
  const match = roiId.match(/^roi_(\d+)$/i);
  if (!match) {
    return null;
  }
  const index = Number(match[1]) - 1;
  return Number.isFinite(index) && index >= 0 ? index : null;
}

function selectedRoi() {
  if (state.selected.type !== "roi") {
    return null;
  }
  return state.rois[state.selected.index] || null;
}

function selectedPoint() {
  if (state.selected.type !== "point") {
    return null;
  }
  return state.points[state.selected.index] || null;
}

function selectedLine() {
  if (state.selected.type !== "line") {
    return null;
  }
  return state.lines[state.selected.index] || null;
}

function getViewportSize() {
  const styles = window.getComputedStyle(canvasWrap);
  const width = canvasWrap.clientWidth - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight);
  const height = canvasWrap.clientHeight - parseFloat(styles.paddingTop) - parseFloat(styles.paddingBottom);
  return {
    width: Math.max(320, Math.round(width)),
    height: Math.max(320, Math.round(height)),
  };
}

function getViewScale() {
  return state.fitScale * state.zoom;
}

function imageToCanvas(x, y) {
  const viewScale = getViewScale();
  return [x * viewScale + state.offsetX, y * viewScale + state.offsetY];
}

function pointColor(kind) {
  return (POINT_KIND_META[kind] || POINT_KIND_META.mark).color;
}

function pointShort(kind) {
  return (POINT_KIND_META[kind] || POINT_KIND_META.mark).short;
}

function lineColor(kind) {
  return (LINE_KIND_META[kind] || LINE_KIND_META.custom).color;
}

function lineShort(kind) {
  return (LINE_KIND_META[kind] || LINE_KIND_META.custom).short;
}

function nextPointLabel(kind) {
  const count = state.points.filter((point) => point.kind === kind).length + 1;
  return `${pointShort(kind)}_${String(count).padStart(2, "0")}`;
}

function nextLineLabel(kind) {
  const count = state.lines.filter((line) => line.kind === kind).length + 1;
  return `${lineShort(kind)}_${String(count).padStart(2, "0")}`;
}

function syncToolButtons() {
  toolRoiBtn.classList.toggle("active", state.tool === "roi");
  toolPointBtn.classList.toggle("active", state.tool === "point");
  toolLineBtn.classList.toggle("active", state.tool === "line");
}

function syncPointDraftLabel(force = false) {
  if (!force && pointLabelInput.value.trim()) {
    return;
  }
  pointLabelInput.value = nextPointLabel(state.pointKind);
}

function syncLineDraftLabel(force = false) {
  if (!force && lineLabelInput.value.trim()) {
    return;
  }
  lineLabelInput.value = nextLineLabel(state.lineKind);
}

function parseRealDistance(value) {
  if (value == null || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatRealDistance(value) {
  return value == null || !Number.isFinite(Number(value)) ? "" : String(Number(value));
}

function normalizeLineGeometry(kind, x1, y1, x2, y2, anchor = "start") {
  let xa = Math.round(Number(x1 || 0));
  let ya = Math.round(Number(y1 || 0));
  let xb = Math.round(Number(x2 || 0));
  let yb = Math.round(Number(y2 || 0));

  if (kind === "vertical_ref") {
    const xRef = anchor === "end" ? xb : xa;
    xa = xRef;
    xb = xRef;
  }

  return { x1: xa, y1: ya, x2: xb, y2: yb };
}

function normalizeRect(x1, y1, x2, y2) {
  const xa = Math.max(0, Math.min(x1, x2));
  const xb = Math.min(state.image.width, Math.max(x1, x2));
  const ya = Math.max(0, Math.min(y1, y2));
  const yb = Math.min(state.image.height, Math.max(y1, y2));
  return [Math.round(xa), Math.round(ya), Math.round(xb), Math.round(yb)];
}

function normalizeRoi(roi) {
  return {
    xyxy: Array.isArray(roi.xyxy) ? roi.xyxy.map((value) => Math.round(Number(value))) : [0, 0, 0, 0],
    note: typeof roi.note === "string" ? roi.note : "",
  };
}

function normalizePoint(point) {
  return {
    kind: typeof point.kind === "string" ? point.kind : "mark",
    label: typeof point.label === "string" ? point.label : "",
    x: Math.round(Number(point.x || 0)),
    y: Math.round(Number(point.y || 0)),
    roiIndex: roiIndexFromId(point.roi_id),
    note: typeof point.note === "string" ? point.note : "",
  };
}

function normalizeLine(line) {
  const kind = typeof line.kind === "string" ? line.kind : "horizontal_ref";
  const geometry = normalizeLineGeometry(kind, line.x1, line.y1, line.x2, line.y2);
  return {
    kind,
    label: typeof line.label === "string" ? line.label : "",
    x1: geometry.x1,
    y1: geometry.y1,
    x2: geometry.x2,
    y2: geometry.y2,
    roiIndex: roiIndexFromId(line.roi_id),
    note: typeof line.note === "string" ? line.note : "",
    realDistance: parseRealDistance(line.real_distance ?? line.realDistance),
  };
}

function normalizeSrcPointsOverride(points) {
  if (!Array.isArray(points) || points.length !== 4) {
    return null;
  }
  const normalized = points.map((point) => {
    if (Array.isArray(point)) {
      return [Math.round(Number(point[0])), Math.round(Number(point[1]))];
    }
    return [Math.round(Number(point.x)), Math.round(Number(point.y))];
  });
  return normalized.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y)) ? normalized : null;
}

function normalizeDstRectOverride(rect) {
  if (!Array.isArray(rect) || rect.length !== 4) {
    return null;
  }
  const normalized = rect.map((value) => Number(value));
  return normalized.every((value) => Number.isFinite(value)) ? normalized : null;
}

function currentHomographySrcPoints() {
  if (Array.isArray(state.homographyQuad) && state.homographyQuad.length === 4) {
    return normalizeSrcPointsOverride(state.homographyQuad);
  }
  return normalizeSrcPointsOverride(state.srcPointsOverride);
}

function clampOffset() {
  if (!state.image) {
    return;
  }

  const viewport = getViewportSize();
  const viewScale = getViewScale();
  const scaledWidth = state.image.width * viewScale;
  const scaledHeight = state.image.height * viewScale;

  if (scaledWidth <= viewport.width) {
    state.offsetX = Math.round((viewport.width - scaledWidth) / 2);
  } else {
    const minX = viewport.width - scaledWidth;
    state.offsetX = Math.max(minX, Math.min(0, state.offsetX));
  }

  if (scaledHeight <= viewport.height) {
    state.offsetY = Math.round((viewport.height - scaledHeight) / 2);
  } else {
    const minY = viewport.height - scaledHeight;
    state.offsetY = Math.max(minY, Math.min(0, state.offsetY));
  }
}

function fitImage(resetZoom = false) {
  if (!state.image) {
    return;
  }

  const viewport = getViewportSize();
  state.fitScale = Math.min(viewport.width / state.image.width, viewport.height / state.image.height);
  if (resetZoom) {
    state.zoom = 1;
  }
  clampOffset();
}

function pointerToImage(event) {
  const rect = roiCanvas.getBoundingClientRect();
  const canvasX = event.clientX - rect.left;
  const canvasY = event.clientY - rect.top;
  const viewScale = getViewScale();
  return [
    Math.max(0, Math.min(state.image.width, (canvasX - state.offsetX) / viewScale)),
    Math.max(0, Math.min(state.image.height, (canvasY - state.offsetY) / viewScale)),
  ];
}

function pointHitTest(clientX, clientY) {
  if (!state.image) {
    return null;
  }

  const rect = roiCanvas.getBoundingClientRect();
  const canvasX = clientX - rect.left;
  const canvasY = clientY - rect.top;
  let best = null;

  state.points.forEach((point, index) => {
    const [px, py] = imageToCanvas(point.x, point.y);
    const distance = Math.hypot(px - canvasX, py - canvasY);
    if (distance <= 14 && (!best || distance < best.distance)) {
      best = { index, distance };
    }
  });

  return best ? best.index : null;
}

function distanceToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (dx === 0 && dy === 0) {
    return Math.hypot(px - x1, py - y1);
  }

  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
  const projX = x1 + t * dx;
  const projY = y1 + t * dy;
  return Math.hypot(px - projX, py - projY);
}

function lineHitTest(clientX, clientY) {
  if (!state.image) {
    return null;
  }

  const rect = roiCanvas.getBoundingClientRect();
  const canvasX = clientX - rect.left;
  const canvasY = clientY - rect.top;
  let best = null;

  state.lines.forEach((line, index) => {
    const [cx1, cy1] = imageToCanvas(line.x1, line.y1);
    const [cx2, cy2] = imageToCanvas(line.x2, line.y2);
    const distance = distanceToSegment(canvasX, canvasY, cx1, cy1, cx2, cy2);
    if (distance <= 10 && (!best || distance < best.distance)) {
      best = { index, distance };
    }
  });

  return best ? best.index : null;
}

function lineEndpointHitTest(clientX, clientY) {
  if (!state.image) {
    return null;
  }

  const rect = roiCanvas.getBoundingClientRect();
  const canvasX = clientX - rect.left;
  const canvasY = clientY - rect.top;
  let best = null;

  state.lines.forEach((line, index) => {
    [
      { key: "start", x: line.x1, y: line.y1 },
      { key: "end", x: line.x2, y: line.y2 },
    ].forEach((endpoint) => {
      const [px, py] = imageToCanvas(endpoint.x, endpoint.y);
      const distance = Math.hypot(px - canvasX, py - canvasY);
      if (distance <= 14 && (!best || distance < best.distance)) {
        best = { index, endpoint: endpoint.key, distance };
      }
    });
  });

  return best;
}

function roiHitTest(x, y) {
  for (let i = state.rois.length - 1; i >= 0; i -= 1) {
    const [x1, y1, x2, y2] = state.rois[i].xyxy;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      return i;
    }
  }
  return null;
}

function setCanvasCursor() {
  roiCanvas.classList.toggle("panning", state.dragMode === "pan");
}

function drawLineAnnotation(line, index, isSelected, options = {}) {
  const { dashed = false } = options;
  const [x1, y1] = imageToCanvas(line.x1, line.y1);
  const [x2, y2] = imageToCanvas(line.x2, line.y2);
  const color = lineColor(line.kind);
  const label = line.label || `${lineShort(line.kind)}_${String(index + 1).padStart(2, "0")}`;

  roiCtx.save();
  roiCtx.strokeStyle = color;
  roiCtx.fillStyle = color;
  roiCtx.lineWidth = isSelected ? 3 : 2;
  if (dashed) {
    roiCtx.setLineDash([9, 5]);
  }

  roiCtx.beginPath();
  roiCtx.moveTo(x1, y1);
  roiCtx.lineTo(x2, y2);
  roiCtx.stroke();
  roiCtx.setLineDash([]);

  roiCtx.beginPath();
  roiCtx.arc(x1, y1, isSelected ? 6 : 4.5, 0, Math.PI * 2);
  roiCtx.arc(x2, y2, isSelected ? 6 : 4.5, 0, Math.PI * 2);
  roiCtx.fill();

  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  roiCtx.font = '700 13px "Space Grotesk", sans-serif';
  roiCtx.fillText(label, midX + 10, midY - 10);
  roiCtx.restore();
}

function drawCanvas() {
  if (!state.image) {
    return;
  }

  const viewport = getViewportSize();
  roiCanvas.width = viewport.width;
  roiCanvas.height = viewport.height;

  roiCtx.clearRect(0, 0, viewport.width, viewport.height);
  roiCtx.fillStyle = "#030506";
  roiCtx.fillRect(0, 0, viewport.width, viewport.height);

  const viewScale = getViewScale();
  const drawWidth = state.image.width * viewScale;
  const drawHeight = state.image.height * viewScale;
  roiCtx.drawImage(state.image, state.offsetX, state.offsetY, drawWidth, drawHeight);

  state.rois.forEach((roi, index) => {
    const [x1, y1] = imageToCanvas(roi.xyxy[0], roi.xyxy[1]);
    const [x2, y2] = imageToCanvas(roi.xyxy[2], roi.xyxy[3]);
    const isSelected = state.selected.type === "roi" && state.selected.index === index;

    roiCtx.lineWidth = isSelected ? 3 : 2;
    roiCtx.strokeStyle = isSelected ? "#22d4c0" : "#f3bc5d";
    roiCtx.fillStyle = isSelected ? "rgba(34, 212, 192, 0.12)" : "rgba(243, 188, 93, 0.08)";
    roiCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
    roiCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    roiCtx.fillStyle = isSelected ? "#22d4c0" : "#f3bc5d";
    roiCtx.font = '700 14px "Space Grotesk", sans-serif';
    roiCtx.fillText(`ROI ${index + 1}`, x1 + 10, Math.max(20, y1 - 8));
  });

  state.lines.forEach((line, index) => {
    drawLineAnnotation(line, index, state.selected.type === "line" && state.selected.index === index);
  });

  state.points.forEach((point, index) => {
    const [px, py] = imageToCanvas(point.x, point.y);
    const isSelected = state.selected.type === "point" && state.selected.index === index;
    const color = pointColor(point.kind);
    const label = point.label || `${pointShort(point.kind)}_${String(index + 1).padStart(2, "0")}`;

    roiCtx.fillStyle = color;
    roiCtx.strokeStyle = color;
    roiCtx.lineWidth = isSelected ? 3 : 2;
    roiCtx.beginPath();
    roiCtx.arc(px, py, isSelected ? 7 : 5, 0, Math.PI * 2);
    roiCtx.fill();

    roiCtx.beginPath();
    roiCtx.arc(px, py, isSelected ? 14 : 10, 0, Math.PI * 2);
    roiCtx.stroke();

    roiCtx.font = '700 13px "Space Grotesk", sans-serif';
    roiCtx.fillText(label, px + 12, py - 10);
  });

  if (state.pendingLineStart) {
    const hover = state.pendingLineHover || state.pendingLineStart;
    const previewLine = normalizeLine({
      kind: state.lineKind,
      label: lineLabelInput.value.trim() || nextLineLabel(state.lineKind),
      x1: state.pendingLineStart.x,
      y1: state.pendingLineStart.y,
      x2: hover.x,
      y2: hover.y,
      real_distance: parseRealDistance(lineRealDistanceInput?.value),
    });
    drawLineAnnotation(
      previewLine,
      state.lines.length,
      true,
      { dashed: true },
    );
  }

  if (state.draft) {
    const [x1, y1] = imageToCanvas(state.draft[0], state.draft[1]);
    const [x2, y2] = imageToCanvas(state.draft[2], state.draft[3]);
    roiCtx.lineWidth = 2;
    roiCtx.strokeStyle = "#4dd6ff";
    roiCtx.setLineDash([8, 4]);
    roiCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    roiCtx.setLineDash([]);
  }

  setCanvasCursor();
}

function buildAnnotationCard(title, pillClass, pillText, metaText, noteText, isSelected, onClick) {
  const item = document.createElement("div");
  item.className = `annotation-item ${isSelected ? "selected" : ""}`;

  const head = document.createElement("div");
  head.className = "annotation-item-head";

  const titleEl = document.createElement("strong");
  titleEl.textContent = title;
  head.appendChild(titleEl);

  const pill = document.createElement("span");
  pill.className = `annotation-pill ${pillClass}`;
  pill.textContent = pillText;
  head.appendChild(pill);

  const meta = document.createElement("div");
  meta.className = "annotation-meta";
  meta.textContent = metaText;

  item.appendChild(head);
  item.appendChild(meta);

  if (noteText && noteText.trim()) {
    const preview = document.createElement("div");
    preview.className = "annotation-note-preview";
    preview.textContent = noteText.length > 120 ? `${noteText.slice(0, 120)}...` : noteText;
    item.appendChild(preview);
  }

  item.addEventListener("click", onClick);
  return item;
}

function rerenderAllAnnotations() {
  renderRoiList();
  renderPointList();
  renderLineList();
  renderSelection();
  drawCanvas();
  updateDashboard();
}

function renderRoiList() {
  roiList.innerHTML = "";

  if (state.rois.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Todavia no hay ROIs. Usa modo ROI y arrastra sobre la imagen.";
    roiList.appendChild(empty);
    return;
  }

  state.rois.forEach((roi, index) => {
    const width = roi.xyxy[2] - roi.xyxy[0];
    const height = roi.xyxy[3] - roi.xyxy[1];
    roiList.appendChild(
      buildAnnotationCard(
        `ROI ${index + 1}`,
        "roi",
        `${width} x ${height}`,
        `xyxy: ${roi.xyxy.join(", ")}`,
        roi.note,
        state.selected.type === "roi" && state.selected.index === index,
        () => {
          state.selected = { type: "roi", index };
          renderSelection();
          renderRoiList();
          renderPointList();
          drawCanvas();
          updateDashboard();
        },
      ),
    );
  });
}

function renderPointList() {
  pointList.innerHTML = "";

  if (state.points.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Todavia no hay puntos. Cambia a modo Points y haz click para agregarlos.";
    pointList.appendChild(empty);
    return;
  }

  state.points.forEach((point, index) => {
    const roiLink = point.roiIndex == null ? "sin ROI" : `ROI ${point.roiIndex + 1}`;
    pointList.appendChild(
      buildAnnotationCard(
        point.label || `point_${String(index + 1).padStart(2, "0")}`,
        point.kind,
        point.kind,
        `x=${point.x}, y=${point.y} | ${roiLink}`,
        point.note,
        state.selected.type === "point" && state.selected.index === index,
        () => {
          state.selected = { type: "point", index };
          renderSelection();
          renderRoiList();
          renderPointList();
          drawCanvas();
          updateDashboard();
        },
      ),
    );
  });
}

function renderLineList() {
  lineList.innerHTML = "";

  if (state.lines.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Todavia no hay lineas. Cambia a modo Lines y usa dos clicks.";
    lineList.appendChild(empty);
    return;
  }

  state.lines.forEach((line, index) => {
    const roiLink = line.roiIndex == null ? "sin ROI" : `ROI ${line.roiIndex + 1}`;
    const length = Math.round(Math.hypot(line.x2 - line.x1, line.y2 - line.y1));
    const refText = line.realDistance == null ? "" : ` | ref=${formatRealDistance(line.realDistance)}`;
    lineList.appendChild(
      buildAnnotationCard(
        line.label || `line_${String(index + 1).padStart(2, "0")}`,
        line.kind,
        line.kind,
        `(${line.x1}, ${line.y1}) -> (${line.x2}, ${line.y2}) | len=${length}${refText} | ${roiLink}`,
        line.note,
        state.selected.type === "line" && state.selected.index === index,
        () => {
          state.selected = { type: "line", index };
          renderSelection();
          renderRoiList();
          renderPointList();
          renderLineList();
          drawCanvas();
          updateDashboard();
        },
      ),
    );
  });
}

function drawPointPreview(point) {
  cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
  const halfWindow = 70;
  const srcX = Math.max(0, point.x - halfWindow);
  const srcY = Math.max(0, point.y - halfWindow);
  const srcW = Math.min(state.image.width - srcX, halfWindow * 2);
  const srcH = Math.min(state.image.height - srcY, halfWindow * 2);
  cropCtx.drawImage(state.image, srcX, srcY, srcW, srcH, 0, 0, cropCanvas.width, cropCanvas.height);

  const markerX = ((point.x - srcX) / srcW) * cropCanvas.width;
  const markerY = ((point.y - srcY) / srcH) * cropCanvas.height;
  cropCtx.strokeStyle = pointColor(point.kind);
  cropCtx.lineWidth = 2;
  cropCtx.beginPath();
  cropCtx.moveTo(markerX - 16, markerY);
  cropCtx.lineTo(markerX + 16, markerY);
  cropCtx.moveTo(markerX, markerY - 16);
  cropCtx.lineTo(markerX, markerY + 16);
  cropCtx.stroke();
}

function drawRoiPreview(roi) {
  cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
  const [x1, y1, x2, y2] = roi.xyxy;
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  const scale = Math.min(cropCanvas.width / width, cropCanvas.height / height, 1);
  const drawWidth = Math.round(width * scale);
  const drawHeight = Math.round(height * scale);
  const dx = Math.round((cropCanvas.width - drawWidth) / 2);
  const dy = Math.round((cropCanvas.height - drawHeight) / 2);
  cropCtx.drawImage(state.image, x1, y1, width, height, dx, dy, drawWidth, drawHeight);
}

function drawLinePreview(line) {
  cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
  const padding = 40;
  const srcX = Math.max(0, Math.min(line.x1, line.x2) - padding);
  const srcY = Math.max(0, Math.min(line.y1, line.y2) - padding);
  const srcW = Math.max(1, Math.min(state.image.width - srcX, Math.abs(line.x2 - line.x1) + padding * 2));
  const srcH = Math.max(1, Math.min(state.image.height - srcY, Math.abs(line.y2 - line.y1) + padding * 2));
  cropCtx.drawImage(state.image, srcX, srcY, srcW, srcH, 0, 0, cropCanvas.width, cropCanvas.height);

  const scaleX = cropCanvas.width / srcW;
  const scaleY = cropCanvas.height / srcH;
  const x1 = (line.x1 - srcX) * scaleX;
  const y1 = (line.y1 - srcY) * scaleY;
  const x2 = (line.x2 - srcX) * scaleX;
  const y2 = (line.y2 - srcY) * scaleY;
  cropCtx.strokeStyle = lineColor(line.kind);
  cropCtx.fillStyle = lineColor(line.kind);
  cropCtx.lineWidth = 2;
  cropCtx.beginPath();
  cropCtx.moveTo(x1, y1);
  cropCtx.lineTo(x2, y2);
  cropCtx.stroke();
  cropCtx.beginPath();
  cropCtx.arc(x1, y1, 5, 0, Math.PI * 2);
  cropCtx.arc(x2, y2, 5, 0, Math.PI * 2);
  cropCtx.fill();
}

function renderSelection() {
  cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);

  if (!state.image || !state.selected.type) {
    selectedMeta.textContent = "Nada seleccionado.";
    selectedLabelInput.value = "";
    selectedLabelInput.disabled = true;
    selectedLineRealDistanceInput.value = "";
    selectedLineRealDistanceInput.disabled = true;
    annotationNoteInput.value = "";
    annotationNoteInput.disabled = true;
    annotationHint.textContent = "Selecciona un ROI, punto o linea para editarlo.";
    return;
  }

  if (state.selected.type === "roi") {
    const roi = selectedRoi();
    if (!roi) {
      return;
    }
    const [x1, y1, x2, y2] = roi.xyxy;
    selectedMeta.textContent =
      `ROI ${state.selected.index + 1}\n` +
      `xyxy: ${x1}, ${y1}, ${x2}, ${y2}\n` +
      `size: ${x2 - x1} x ${y2 - y1}`;
    drawRoiPreview(roi);
    selectedLabelInput.value = "";
    selectedLabelInput.disabled = true;
    selectedLineRealDistanceInput.value = "";
    selectedLineRealDistanceInput.disabled = true;
    annotationNoteInput.disabled = false;
    annotationNoteInput.value = roi.note || "";
    annotationHint.textContent = "El ROI solo guarda nota. Puntos y lineas guardan label y nota.";
    return;
  }

  if (state.selected.type === "point") {
    const point = selectedPoint();
    if (!point) {
      return;
    }

    const roiLink = point.roiIndex == null ? "sin ROI" : `ROI ${point.roiIndex + 1}`;
    selectedMeta.textContent =
      `${point.label || "point"}\n` +
      `kind: ${point.kind}\n` +
      `xy: ${point.x}, ${point.y}\n` +
      `link: ${roiLink}`;
    drawPointPreview(point);
    selectedLabelInput.disabled = false;
    selectedLabelInput.value = point.label || "";
    selectedLineRealDistanceInput.value = "";
    selectedLineRealDistanceInput.disabled = true;
    annotationNoteInput.disabled = false;
    annotationNoteInput.value = point.note || "";
    annotationHint.textContent = "Edita label y nota del punto seleccionado. Se guardan en TOML.";
    return;
  }

  const line = selectedLine();
  if (!line) {
    return;
  }

  const roiLink = line.roiIndex == null ? "sin ROI" : `ROI ${line.roiIndex + 1}`;
  const length = Math.round(Math.hypot(line.x2 - line.x1, line.y2 - line.y1));
  selectedMeta.textContent =
    `${line.label || "line"}\n` +
    `kind: ${line.kind}\n` +
    `p1: ${line.x1}, ${line.y1}\n` +
    `p2: ${line.x2}, ${line.y2}\n` +
    `len: ${length}\n` +
    `ref: ${line.realDistance == null ? "-" : formatRealDistance(line.realDistance)}\n` +
    `link: ${roiLink}`;
  drawLinePreview(line);
  selectedLabelInput.disabled = false;
  selectedLabelInput.value = line.label || "";
  selectedLineRealDistanceInput.disabled = false;
  selectedLineRealDistanceInput.value = formatRealDistance(line.realDistance);
  annotationNoteInput.disabled = false;
  annotationNoteInput.value = line.note || "";
  annotationHint.textContent = "Edita label, distancia real y nota de la linea seleccionada. Se guardan en TOML.";
}

function updateDashboard() {
  currentImageName.textContent = state.imageName || "-";
  imageMeta.textContent = state.image ? `${state.image.width} x ${state.image.height}` : "-";
  annotationHero.textContent = `${state.rois.length} ROIs / ${state.points.length} puntos / ${state.lines.length} lineas`;
  roiCountBadge.textContent = String(state.rois.length);
  pointCountBadge.textContent = String(state.points.length);
  lineCountBadge.textContent = String(state.lines.length);

  if (!state.image) {
    canvasMeta.textContent = "Sin imagen";
    return;
  }

  const selectionText =
    state.selected.type === "roi"
      ? `ROI ${state.selected.index + 1}`
      : state.selected.type === "point"
        ? `${selectedPoint()?.label || "point"}`
        : state.selected.type === "line"
          ? `${selectedLine()?.label || "line"}`
          : "sin seleccion";
  const effectiveSelectionText = state.pendingLineStart
    ? `linea pendiente desde ${state.pendingLineStart.x}, ${state.pendingLineStart.y}`
    : selectionText;
  canvasMeta.textContent = `Tool ${state.tool} | Zoom ${getZoomText()} | ${effectiveSelectionText}`;
}

function serializeRois() {
  return state.rois.map((roi) => ({
    xyxy: roi.xyxy,
    note: roi.note || "",
  }));
}

function serializePoints() {
  return state.points.map((point, index) => ({
    id: `point_${String(index + 1).padStart(2, "0")}`,
    kind: point.kind,
    label: point.label || nextPointLabel(point.kind),
    x: point.x,
    y: point.y,
    roi_id: roiIdFromIndex(point.roiIndex),
    note: point.note || "",
  }));
}

function serializeLines() {
  return state.lines.map((line, index) => ({
    id: `line_${String(index + 1).padStart(2, "0")}`,
    kind: line.kind,
    label: line.label || nextLineLabel(line.kind),
    x1: line.x1,
    y1: line.y1,
    x2: line.x2,
    y2: line.y2,
    roi_id: roiIdFromIndex(line.roiIndex),
    note: line.note || "",
    real_distance: line.realDistance,
  }));
}

function buildTomlExport() {
  const lines = [];
  lines.push(`image_name = ${JSON.stringify(state.imageName || "")}`);
  lines.push(`updated_at = ${JSON.stringify(new Date().toISOString().slice(0, 19))}`);

  const srcPointsOverride = currentHomographySrcPoints();
  if (srcPointsOverride) {
    const pointsText = srcPointsOverride.map(([x, y]) => `[${x}, ${y}]`).join(", ");
    lines.push(`src_points_override = [${pointsText}]`);
  }

  const dstRectOverride = srcPointsOverride ? null : Array.isArray(state.previewRect) ? state.previewRect : state.dstRectOverride;
  if (Array.isArray(dstRectOverride) && dstRectOverride.length === 4) {
    lines.push(`dst_rect_override = [${dstRectOverride.map((value) => Number(value).toFixed(3).replace(/\.000$/, "")).join(", ")}]`);
  }

  serializeRois().forEach((roi, index) => {
    lines.push("");
    lines.push("[[rois]]");
    lines.push(`id = ${JSON.stringify(roiIdFromIndex(index))}`);
    lines.push(`xyxy = [${roi.xyxy.join(", ")}]`);
    lines.push(`note = ${JSON.stringify(roi.note || "")}`);
  });

  serializePoints().forEach((point) => {
    lines.push("");
    lines.push("[[points]]");
    lines.push(`id = ${JSON.stringify(point.id)}`);
    lines.push(`kind = ${JSON.stringify(point.kind)}`);
    lines.push(`label = ${JSON.stringify(point.label || "")}`);
    lines.push(`x = ${point.x}`);
    lines.push(`y = ${point.y}`);
    lines.push(`roi_id = ${JSON.stringify(point.roi_id || "")}`);
    lines.push(`note = ${JSON.stringify(point.note || "")}`);
  });

  serializeLines().forEach((line) => {
    lines.push("");
    lines.push("[[lines]]");
    lines.push(`id = ${JSON.stringify(line.id)}`);
    lines.push(`kind = ${JSON.stringify(line.kind)}`);
    lines.push(`label = ${JSON.stringify(line.label || "")}`);
    lines.push(`x1 = ${line.x1}`);
    lines.push(`y1 = ${line.y1}`);
    lines.push(`x2 = ${line.x2}`);
    lines.push(`y2 = ${line.y2}`);
    lines.push(`roi_id = ${JSON.stringify(line.roi_id || "")}`);
    lines.push(`note = ${JSON.stringify(line.note || "")}`);
    if (line.real_distance != null) {
      lines.push(`real_distance = ${Number(line.real_distance)}`);
    }
  });

  return `${lines.join("\n")}\n`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let details = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload && payload.error) {
        details = `${response.status}: ${payload.error}`;
      }
    } catch (_) {
      // ignore non-json error bodies
    }
    throw new Error(details);
  }
  return await response.json();
}

async function loadImageAndAnnotations(name) {
  state.imageName = name;
  setStatus(`Loading ${name}...`);
  closeHomographyPreview();
  updateDashboard();

  const imageUrl = imageUrlByName(name);
  if (!imageUrl) {
    throw new Error(`Image URL not found for ${name}`);
  }

  const img = new Image();
  img.src = `${imageUrl}?_ts=${Date.now()}`;
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
  });
  state.image = img;
  fitImage(true);

  const data = await fetchJson(`/api/rois?image=${encodeURIComponent(name)}`);
  state.rois = (data.rois || []).map(normalizeRoi);
  state.points = (data.points || []).map(normalizePoint);
  state.lines = (data.lines || []).map(normalizeLine);
  state.srcPointsOverride = normalizeSrcPointsOverride(data.src_points_override);
  state.dstRectOverride = normalizeDstRectOverride(data.dst_rect_override);
  state.selected = { type: null, index: null };
  state.draft = null;
  state.dragMode = null;
  state.panStartOffset = null;
  state.pendingLineStart = null;
  state.pendingLineHover = null;
  syncPointDraftLabel(true);
  syncLineDraftLabel(true);
  rerenderAllAnnotations();
  setStatus(`Loaded ${name}. ${state.rois.length} ROIs, ${state.points.length} points and ${state.lines.length} lines in storage.`);
}

async function saveAnnotations() {
  if (!state.imageName) {
    return;
  }

  setStatus(`Saving ${state.rois.length} ROIs, ${state.points.length} points and ${state.lines.length} lines...`);
  const payload = await fetchJson("/api/rois", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_name: state.imageName,
      rois: serializeRois(),
      points: serializePoints(),
      lines: serializeLines(),
      src_points_override: currentHomographySrcPoints(),
      dst_rect_override: currentHomographySrcPoints()
        ? null
        : Array.isArray(state.previewRect)
          ? state.previewRect
          : state.dstRectOverride,
    }),
  });

  state.rois = (payload.rois || []).map(normalizeRoi);
  state.points = (payload.points || []).map(normalizePoint);
  state.lines = (payload.lines || []).map(normalizeLine);
  state.srcPointsOverride = normalizeSrcPointsOverride(payload.src_points_override);
  state.dstRectOverride = normalizeDstRectOverride(payload.dst_rect_override);
  rerenderAllAnnotations();
  setStatus(`Saved to manual_rois/${state.imageName.replace(/\.jpg$/i, "_rois.toml")}`);
}

function currentAnnotationsPayload(imageName = state.imageName) {
  const srcPointsOverride = currentHomographySrcPoints();
  return {
    image_name: imageName,
    rois: serializeRois(),
    points: serializePoints(),
    lines: serializeLines(),
    src_points_override: srcPointsOverride,
    dst_rect_override: srcPointsOverride
      ? null
      : Array.isArray(state.previewRect)
        ? state.previewRect
        : state.dstRectOverride,
  };
}

function closeHomographyPreview() {
  state.homographyQuad = null;
  state.overlayPreviewDrag = null;
  state.previewRect = null;
  state.previewHomographyMatrix = null;
  state.previewInverseHomographyMatrix = null;
  state.previewBaseSize = null;
  state.previewMode = "homography";
  clearPreviewViewer(previewViewers.overlay);
  clearPreviewViewer(previewViewers.warp);
  previewMeta.textContent = "";
}

function openHomographyPreview(data) {
  state.previewRect = Array.isArray(data.dst_rect)
    ? data.dst_rect.map((value) => Number(value))
    : null;
  state.previewHomographyMatrix = Array.isArray(data.homography_matrix) ? data.homography_matrix : null;
  state.previewInverseHomographyMatrix = Array.isArray(data.inverse_homography_matrix)
    ? data.inverse_homography_matrix
    : null;
  state.previewBaseSize = Array.isArray(data.base_size) ? data.base_size.map((value) => Number(value)) : null;
  state.homographyQuad = Array.isArray(data.src_points)
    ? data.src_points.map((point) => ({ x: Number(point[0]), y: Number(point[1]) }))
    : null;
  const stamp = Date.now();
  previewOverlayImg.src = `${data.overlay_url}?_ts=${stamp}`;
  previewWarpImg.src = `${data.warp_url}?_ts=${stamp}`;
  if (previewOverlayImg.complete) {
    resetPreviewViewer(previewViewers.overlay);
  }
  if (previewWarpImg.complete) {
    resetPreviewViewer(previewViewers.warp);
  }
  const labels = (data.used_line_labels || []).join(" / ");
  const size = Array.isArray(data.output_size) ? `${data.output_size[0]} x ${data.output_size[1]}` : "-";
  const rectText =
    Array.isArray(state.previewRect) && state.previewRect.length === 4
      ? `rect ${state.previewRect.map((value) => Math.round(value)).join(", ")}`
      : "rect -";
  previewMeta.textContent = `Lineas usadas: ${labels || "-"} | salida: ${size} | ${rectText}`;
  drawHomographyQuadOverlay();
}

async function previewHomography() {
  if (!state.imageName) {
    return;
  }

  state.previewMode = "homography";
  previewMeta.textContent = "";
  setStatus("Construyendo preview de homografia...");
  const data = await fetchJson("/api/homography_preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentAnnotationsPayload()),
  });
  openHomographyPreview(data);
  setStatus("Preview de homografia listo.");
}

async function previewTubeDetection() {
  if (!state.imageName) {
    return;
  }

  state.previewMode = "tubes";
  previewMeta.textContent = "";
  setStatus("Detectando tubos sobre el warp actual...");
  const data = await fetchJson("/api/tube_detection_preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentAnnotationsPayload()),
  });
  openHomographyPreview(data);
  const periodText =
    data.dominant_period == null ? "periodo=-" : `periodo=${Number(data.dominant_period).toFixed(1)}px`;
  const roiText =
    Array.isArray(data.detection_roi) && data.detection_roi.length === 4
      ? `roi=${data.detection_roi.map((value) => Math.round(Number(value))).join(",")}`
      : "roi=warp";
  const scaleText =
    data.px_per_in == null ? "scale=-" : `scale=${Number(data.px_per_in).toFixed(2)}px/in`;
  const refText = Array.isArray(data.reference_lines) ? `refs=${data.reference_lines.length}` : "refs=0";
  previewMeta.textContent += ` | tubos=${data.tube_count ?? 0} | ${periodText} | ${roiText} | ${scaleText} | ${refText}`;
  setStatus("Preview de tubos listo.");
}

async function propagateAnnotationsToAllImages() {
  if (!state.imageName) {
    return;
  }

  const targets = state.images.filter((item) => item.name !== state.imageName);
  if (targets.length === 0) {
    setStatus("No hay otras imagenes a las que copiar.");
    return;
  }

  setStatus(`Copiando anotaciones a ${targets.length} imagenes...`);
  const basePayload = currentAnnotationsPayload();
  for (const target of targets) {
    await fetchJson("/api/rois", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_name: target.name,
        rois: basePayload.rois,
        points: basePayload.points,
        lines: basePayload.lines,
      }),
    });
  }
  setStatus(`Anotaciones copiadas a ${targets.length} imagenes.`);
}

function downloadCurrentToml() {
  const blob = new Blob([buildTomlExport()], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.imageName.replace(/\.jpg$/i, "")}_rois_export.toml`;
  link.click();
  URL.revokeObjectURL(url);
}

function deleteSelectedAnnotation() {
  if (!state.selected.type) {
    return;
  }

  if (state.selected.type === "point") {
    state.points.splice(state.selected.index, 1);
    state.selected = { type: null, index: null };
    rerenderAllAnnotations();
    return;
  }

  if (state.selected.type === "line") {
    state.lines.splice(state.selected.index, 1);
    state.selected = { type: null, index: null };
    rerenderAllAnnotations();
    return;
  }

  const removedIndex = state.selected.index;
  state.rois.splice(removedIndex, 1);
  state.points = state.points.map((point) => {
    if (point.roiIndex == null) {
      return point;
    }
    if (point.roiIndex === removedIndex) {
      return { ...point, roiIndex: null };
    }
    if (point.roiIndex > removedIndex) {
      return { ...point, roiIndex: point.roiIndex - 1 };
    }
    return point;
  });
  state.lines = state.lines.map((line) => {
    if (line.roiIndex == null) {
      return line;
    }
    if (line.roiIndex === removedIndex) {
      return { ...line, roiIndex: null };
    }
    if (line.roiIndex > removedIndex) {
      return { ...line, roiIndex: line.roiIndex - 1 };
    }
    return line;
  });
  state.selected = { type: null, index: null };
  rerenderAllAnnotations();
}

function cleanupPointer(event) {
  state.dragMode = null;
  state.dragStart = null;
  state.dragTarget = null;
  state.draft = null;
  state.panStartOffset = null;
  if (event && roiCanvas.hasPointerCapture(event.pointerId)) {
    roiCanvas.releasePointerCapture(event.pointerId);
  }
  setCanvasCursor();
}

function activeRoiIndexForPlacement(x, y) {
  if (state.selected.type === "roi") {
    return state.selected.index;
  }
  return roiHitTest(x, y);
}

function createPointAt(x, y) {
  const label = pointLabelInput.value.trim() || nextPointLabel(state.pointKind);
  state.points.push({
    kind: state.pointKind,
    label,
    x: Math.round(x),
    y: Math.round(y),
    roiIndex: activeRoiIndexForPlacement(x, y),
    note: "",
  });
  state.selected = { type: "point", index: state.points.length - 1 };
  syncPointDraftLabel(true);
  rerenderAllAnnotations();
}

function cancelPendingLine() {
  if (!state.pendingLineStart) {
    return;
  }
  state.pendingLineStart = null;
  state.pendingLineHover = null;
  drawCanvas();
  updateDashboard();
  setStatus("Linea cancelada.");
}

function startLineAt(x, y) {
  state.pendingLineStart = {
    x: Math.round(x),
    y: Math.round(y),
    roiIndex: activeRoiIndexForPlacement(x, y),
  };
  state.pendingLineHover = { x: Math.round(x), y: Math.round(y) };
  drawCanvas();
  updateDashboard();
  setStatus("Primer punto de linea fijado. Haz segundo click para cerrarla.");
}

function finishLineAt(x, y) {
  if (!state.pendingLineStart) {
    return;
  }

  const geometry = normalizeLineGeometry(state.lineKind, state.pendingLineStart.x, state.pendingLineStart.y, x, y);
  const length = Math.hypot(geometry.x2 - geometry.x1, geometry.y2 - geometry.y1);
  if (length < 6) {
    setStatus("La linea es demasiado corta. Marca un segundo punto mas separado.");
    return;
  }

  const label = lineLabelInput.value.trim() || nextLineLabel(state.lineKind);
  const roiIndex =
    state.pendingLineStart.roiIndex != null
      ? state.pendingLineStart.roiIndex
      : activeRoiIndexForPlacement(geometry.x2, geometry.y2);
  state.lines.push({
    kind: state.lineKind,
    label,
    x1: geometry.x1,
    y1: geometry.y1,
    x2: geometry.x2,
    y2: geometry.y2,
    roiIndex,
    note: "",
    realDistance: parseRealDistance(lineRealDistanceInput?.value),
  });
  state.selected = { type: "line", index: state.lines.length - 1 };
  state.pendingLineStart = null;
  state.pendingLineHover = null;
  syncLineDraftLabel(true);
  rerenderAllAnnotations();
}

function setTool(tool) {
  state.tool = tool;
  if (tool !== "line") {
    state.pendingLineStart = null;
    state.pendingLineHover = null;
  }
  syncToolButtons();
  updateDashboard();
  drawCanvas();
}

roiCanvas.addEventListener("pointerdown", (event) => {
  if (!state.image) {
    return;
  }

  if (event.button === 1) {
    event.preventDefault();
    state.dragMode = "pan";
    state.dragStart = [event.clientX, event.clientY];
    state.panStartOffset = [state.offsetX, state.offsetY];
    roiCanvas.setPointerCapture(event.pointerId);
    setCanvasCursor();
    return;
  }

  if (event.button !== 0) {
    return;
  }

  const [x, y] = pointerToImage(event);

  const pointIndex = pointHitTest(event.clientX, event.clientY);
  if (pointIndex != null && state.tool !== "line") {
    state.selected = { type: "point", index: pointIndex };
    state.dragMode = "point-move";
    state.dragTarget = { pointIndex };
    roiCanvas.setPointerCapture(event.pointerId);
    renderSelection();
    renderRoiList();
    renderPointList();
    drawCanvas();
    updateDashboard();
    return;
  }

  const endpointHit = lineEndpointHitTest(event.clientX, event.clientY);
  if (endpointHit && (!state.pendingLineStart || state.tool !== "line")) {
    state.selected = { type: "line", index: endpointHit.index };
    state.dragMode = "line-endpoint";
    state.dragTarget = endpointHit;
    roiCanvas.setPointerCapture(event.pointerId);
    renderSelection();
    renderLineList();
    drawCanvas();
    updateDashboard();
    return;
  }

  if (state.tool === "line") {
    if (state.pendingLineStart) {
      finishLineAt(x, y);
      return;
    }
    startLineAt(x, y);
    return;
  }

  const lineIndex = lineHitTest(event.clientX, event.clientY);
  if (lineIndex != null && state.tool !== "point") {
    state.selected = { type: "line", index: lineIndex };
    renderSelection();
    renderLineList();
    drawCanvas();
    updateDashboard();
    return;
  }

  const hitRoiIndex = roiHitTest(x, y);

  if (state.tool === "point") {
    if (hitRoiIndex != null && state.selected.type !== "roi") {
      state.selected = { type: "roi", index: hitRoiIndex };
    }
    createPointAt(x, y);
    return;
  }

  if (hitRoiIndex != null) {
    state.selected = { type: "roi", index: hitRoiIndex };
    renderSelection();
    renderRoiList();
    renderPointList();
    renderLineList();
    drawCanvas();
    updateDashboard();
    return;
  }

  state.dragMode = "draw";
  state.dragStart = [x, y];
  state.draft = [Math.round(x), Math.round(y), Math.round(x), Math.round(y)];
  roiCanvas.setPointerCapture(event.pointerId);
  drawCanvas();
});

roiCanvas.addEventListener("pointermove", (event) => {
  if (state.dragMode === "pan" && state.dragStart && state.panStartOffset) {
    state.offsetX = state.panStartOffset[0] + (event.clientX - state.dragStart[0]);
    state.offsetY = state.panStartOffset[1] + (event.clientY - state.dragStart[1]);
    clampOffset();
    drawCanvas();
    updateDashboard();
    return;
  }

  if (state.dragMode === "point-move" && state.dragTarget) {
    const [x, y] = pointerToImage(event);
    const point = state.points[state.dragTarget.pointIndex];
    if (!point) {
      return;
    }
    point.x = Math.round(x);
    point.y = Math.round(y);
    renderSelection();
    drawCanvas();
    updateDashboard();
    return;
  }

  if (state.dragMode === "line-endpoint" && state.dragTarget) {
    const [x, y] = pointerToImage(event);
    const line = state.lines[state.dragTarget.index];
    if (!line) {
      return;
    }
    if (state.dragTarget.endpoint === "start") {
      const geometry = normalizeLineGeometry(line.kind, x, y, line.x2, line.y2);
      line.x1 = geometry.x1;
      line.y1 = geometry.y1;
      line.x2 = geometry.x2;
      line.y2 = geometry.y2;
    } else {
      const geometry = normalizeLineGeometry(line.kind, line.x1, line.y1, x, y, "end");
      line.x1 = geometry.x1;
      line.y1 = geometry.y1;
      line.x2 = geometry.x2;
      line.y2 = geometry.y2;
    }
    renderSelection();
    drawCanvas();
    updateDashboard();
    return;
  }

  if (state.dragMode === "draw" && state.dragStart) {
    const [x, y] = pointerToImage(event);
    state.draft = normalizeRect(state.dragStart[0], state.dragStart[1], x, y);
    drawCanvas();
    return;
  }

  if (state.tool === "line" && state.pendingLineStart) {
    const [x, y] = pointerToImage(event);
    const geometry = normalizeLineGeometry(state.lineKind, state.pendingLineStart.x, state.pendingLineStart.y, x, y);
    state.pendingLineHover = { x: geometry.x2, y: geometry.y2 };
    drawCanvas();
  }
});

roiCanvas.addEventListener("pointerup", (event) => {
  if (state.dragMode === "pan") {
    cleanupPointer(event);
    return;
  }

  if (state.dragMode === "point-move" || state.dragMode === "line-endpoint") {
    cleanupPointer(event);
    rerenderAllAnnotations();
    return;
  }

  if (state.dragMode !== "draw" || !state.dragStart || !state.draft) {
    cleanupPointer(event);
    return;
  }

  const [x1, y1, x2, y2] = state.draft;
  cleanupPointer(event);
  if (x2 - x1 >= 6 && y2 - y1 >= 6) {
    state.rois.push({ xyxy: [x1, y1, x2, y2], note: "" });
    state.selected = { type: "roi", index: state.rois.length - 1 };
    rerenderAllAnnotations();
  }
});

roiCanvas.addEventListener("pointercancel", (event) => {
  cleanupPointer(event);
  drawCanvas();
});

roiCanvas.addEventListener(
  "wheel",
  (event) => {
    if (!state.image || !event.ctrlKey) {
      return;
    }

    event.preventDefault();
    const rect = roiCanvas.getBoundingClientRect();
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;
    const [imageX, imageY] = pointerToImage(event);
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    const nextZoom = Math.max(state.minZoom, Math.min(state.maxZoom, state.zoom * factor));

    if (nextZoom === state.zoom) {
      return;
    }

    state.zoom = nextZoom;
    const viewScale = getViewScale();
    state.offsetX = canvasX - imageX * viewScale;
    state.offsetY = canvasY - imageY * viewScale;
    clampOffset();
    drawCanvas();
    updateDashboard();
  },
  { passive: false },
);

roiCanvas.addEventListener("mousedown", (event) => {
  if (event.button === 1) {
    event.preventDefault();
  }
});

roiCanvas.addEventListener("auxclick", (event) => {
  if (event.button === 1) {
    event.preventDefault();
  }
});

toolRoiBtn.addEventListener("click", () => setTool("roi"));
toolPointBtn.addEventListener("click", () => setTool("point"));
toolLineBtn.addEventListener("click", () => setTool("line"));

pointKindSelect.addEventListener("change", () => {
  state.pointKind = pointKindSelect.value;
  syncPointDraftLabel(true);
});

pointLabelInput.addEventListener("focus", () => {
  if (!pointLabelInput.value.trim()) {
    syncPointDraftLabel(true);
  }
});

lineKindSelect.addEventListener("change", () => {
  state.lineKind = lineKindSelect.value;
  syncLineDraftLabel(true);
  drawCanvas();
});

lineLabelInput.addEventListener("focus", () => {
  if (!lineLabelInput.value.trim()) {
    syncLineDraftLabel(true);
  }
});

saveBtn.addEventListener("click", () => {
  saveAnnotations().catch((err) => setStatus(String(err)));
});

homographyBtn.addEventListener("click", () => {
  previewHomography().catch((err) => {
    closeHomographyPreview();
    previewMeta.textContent = `Error: ${String(err.message || err)}`;
    setStatus(String(err));
  });
});

tubeDetectionBtn.addEventListener("click", () => {
  previewTubeDetection().catch((err) => {
    previewMeta.textContent = `Error: ${String(err.message || err)}`;
    setStatus(String(err));
  });
});

propagateBtn.addEventListener("click", () => {
  propagateAnnotationsToAllImages().catch((err) => setStatus(String(err)));
});

reloadBtn.addEventListener("click", () => {
  loadImageAndAnnotations(imageSelect.value).catch((err) => setStatus(String(err)));
});

deleteBtn.addEventListener("click", deleteSelectedAnnotation);

clearBtn.addEventListener("click", () => {
  state.rois = [];
  state.points = [];
  state.lines = [];
  state.selected = { type: null, index: null };
  state.pendingLineStart = null;
  state.pendingLineHover = null;
  rerenderAllAnnotations();
});

downloadBtn.addEventListener("click", downloadCurrentToml);
clearPreviewBtn.addEventListener("click", closeHomographyPreview);
toggleRightSidebarBtn.addEventListener("click", () => {
  setRightSidebarCollapsed(!layoutRoot.classList.contains("right-collapsed"));
});

imageSelect.addEventListener("change", () => {
  loadImageAndAnnotations(imageSelect.value).catch((err) => setStatus(String(err)));
});

selectedLabelInput.addEventListener("input", (event) => {
  const value = event.target.value;
  const point = selectedPoint();
  if (point) {
    point.label = value;
    renderSelection();
    renderPointList();
    drawCanvas();
    updateDashboard();
    return;
  }

  const line = selectedLine();
  if (!line) {
    return;
  }
  line.label = value;
  renderSelection();
  renderLineList();
  drawCanvas();
  updateDashboard();
});

selectedLineRealDistanceInput.addEventListener("input", (event) => {
  const line = selectedLine();
  if (!line) {
    return;
  }
  line.realDistance = parseRealDistance(event.target.value);
  renderSelection();
  renderLineList();
  drawCanvas();
  updateDashboard();
});

annotationNoteInput.addEventListener("input", (event) => {
  if (state.selected.type === "roi") {
    const roi = selectedRoi();
    if (!roi) {
      return;
    }
    roi.note = event.target.value;
    renderRoiList();
    return;
  }

  const point = selectedPoint();
  if (point) {
    point.note = event.target.value;
    renderPointList();
    return;
  }

  const line = selectedLine();
  if (!line) {
    return;
  }
  line.note = event.target.value;
  renderLineList();
});

window.addEventListener("keydown", (event) => {
  const typing = isTypingTarget(event.target);

  if (event.key === "Escape") {
    cancelPendingLine();
  }

  if (event.key === "Delete" || event.key === "Backspace") {
    if (!typing) {
      deleteSelectedAnnotation();
    }
  }

  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveAnnotations().catch((err) => setStatus(String(err)));
  }
});

window.addEventListener("resize", () => {
  fitImage(false);
  Object.values(previewViewers).forEach((viewer) => {
    if (hasPreviewImage(viewer) && viewer.img.complete) {
      resetPreviewViewer(viewer);
    }
  });
  drawCanvas();
  updateDashboard();
});

async function loadImagesBootstrap() {
  const data = await fetchJson("/api/images");
  state.images = data.images || [];
  imageSelect.innerHTML = "";
  state.images.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.name;
    imageSelect.appendChild(option);
  });
  if (state.images.length === 0) {
    throw new Error("No images found in test_images.");
  }
  if (!state.imageName) {
    state.imageName = state.images[0].name;
  }
  imageSelect.value = state.imageName;
}

function imageUrlByName(name) {
  const found = state.images.find((item) => item.name === name);
  return found ? found.url : null;
}

async function init() {
  try {
    selectedLabelInput.disabled = true;
    selectedLineRealDistanceInput.disabled = true;
    annotationNoteInput.disabled = true;
    pointKindSelect.value = state.pointKind;
    lineKindSelect.value = state.lineKind;
    syncToolButtons();
    syncPointDraftLabel(true);
    syncLineDraftLabel(true);
    Object.values(previewViewers).forEach(bindPreviewViewer);
    bindOverlayQuadEditor();
    await loadImagesBootstrap();
    await loadImageAndAnnotations(state.imageName);
    setStatus("Ready. Usa ROI para contexto, Points para marcas y Lines para referencias horizontales o verticales.");
  } catch (error) {
    setStatus(String(error));
  }
}

updateDashboard();
init();


